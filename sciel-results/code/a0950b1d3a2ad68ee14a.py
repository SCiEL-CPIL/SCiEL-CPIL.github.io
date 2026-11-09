import time
import numpy as np
from itertools import product

# Global state
_start_time = None
_best_cap_size = 0
_best_priority_cache = None  # Not used for dynamic, but maybe for fallback?
_initialized = False

def _get_all_vectors(n):
    return list(product([0, 1, 2], repeat=n))

def _is_cap_set_addition(v, S):
    """
    Check if v can be added to S without forming a 3-term AP.
    For a 3-term AP (u, v, w), we have 2v = u + w (mod 3).
    So for a fixed v and existing u in S, the required w is (2v - u) mod 3.
    If w is in S, we have a conflict.
    Note: If u == v, w == v. We assume v is not in S yet.
    Also, the AP can be (u, w, v) or (w, u, v) etc. The condition 2v = u + w covers all permutations 
    where v is the middle term. 
    Wait, the definition of a cap set is that it contains no 3-term arithmetic progression.
    A 3-term AP in Z_3^n is a set {a, b, c} such that a + c = 2b.
    So if we add v, we must ensure that for any u, w in S ∪ {v}, we don't have u + w = 2v or u + v = 2w or w + v = 2u.
    
    Actually, the standard check for adding v to S is:
    1. For any u in S, let w = 2v - u. If w in S, then {u, v, w} is an AP.
    2. Also, we must check if v forms an AP with two other elements where v is NOT the middle?
       No, the condition "no 3 distinct elements form an AP" is equivalent to:
       For any distinct u, w in S, is 2u = w + v? No.
       
       Let's stick to the symmetric definition: A set S is a cap set if there are no distinct x, y, z in S such that x + z = 2y.
       When adding v, we need to check if there exist x, y in S such that:
       - x + v = 2y  => v = 2y - x. (v is an endpoint)
       - y + v = 2x  => v = 2x - y. (v is an endpoint)
       - x + y = 2v  => v = (x + y) / 2 = 2(x+y) since 2*2=4=1 mod 3? No, 2 inverse is 2. So v = 2(x+y). (v is middle)
       
       Wait, in Z_3, 2 is the inverse of 2. 
       If x, y, v form an AP, one of them is the middle.
       Case 1: v is middle. x + y = 2v => 2v = x + y. For each x in S, y must be 2v - x. If y in S, conflict.
       Case 2: x is middle. x + v = 2y => 2y = x + v => y = 2(x+v). For each x in S, if y in S, conflict.
       Case 3: y is middle. y + v = 2x => 2x = y + v => x = 2(y+v). For each y in S, if x in S, conflict.
       
       However, due to symmetry, checking Case 1 for all pairs in S is sufficient?
       If x, y, v form an AP, then 2*(middle) = sum of others.
       If v is middle: 2v = x + y. Check: for each x in S, is (2v - x) in S?
       If x is middle: 2x = y + v. This is equivalent to 2v = 2x - y. 
       Is it enough to just check: for each u in S, is (2v - u) in S?
       Let's test. If 2x = y + v, then 2v = 2x - y. Let u = y. Then 2v - u = 2x - y = v? No.
       2v - y = 2x - y - y = 2x - 2y = 2(x-y). This doesn't immediately show it's in S.
       
       Actually, the condition "S is a cap set" is often checked by:
       For all distinct u, v in S, is (u+v)/2 in S? No, that's for the sum.
       
       Let's use the property: S is a cap set iff for all u, v in S, (u+v)/2 is not in S (unless u=v).
       Wait, if u, v, w are distinct and u+w=2v, then w = 2v-u.
       So if we have S, and we want to add v_new.
       We need to ensure no AP is formed involving v_new.
       The APs involving v_new are:
       1. v_new is the middle: u + w = 2*v_new. u, w in S.
          Check: For each u in S, is w = 2*v_new - u in S?
       2. u is the middle: v_new + w = 2*u. w in S.
          Check: For each u in S, is w = 2*u - v_new in S?
       3. w is the middle: v_new + u = 2*w. u in S.
          Check: For each w in S, is u = 2*w - v_new in S?
          
       Note that cases 2 and 3 are symmetric. If there exist u, w in S such that v_new + w = 2u, 
       then let u' = w and w' = u. Then v_new + u' = 2w'. 
       So checking case 2 for all u in S is sufficient for both endpoint cases.
       
       So the full check for adding v is:
       1. For each u in S: if (2*v - u) in S, return False. (v is middle)
       2. For each u in S: if (2*u - v) in S, return False. (u is middle)
       
       Wait, is case 2 covered by case 1?
       If (2*u - v) in S, let w = 2*u - v. Then v + w = 2u.
       This is an AP (v, u, w).
       In case 1, we check if 2*v - u' is in S for some u'.
       This doesn't seem to cover it directly.
       
       So we must check both.
    """
    v_arr = np.array(v)
    for u in S:
        u_arr = np.array(u)
        # Case 1: v is middle. w = 2v - u
        w1 = (2 * v_arr - u_arr) % 3
        if tuple(w1) in S:
            return False
        # Case 2: u is middle. w = 2u - v
        w2 = (2 * u_arr - v_arr) % 3
        if tuple(w2) in S:
            return False
    return True

def _dynamic_greedy(n, time_limit=290):
    """
    Runs a greedy algorithm where the priority of each remaining vector is 
    dynamically adjusted based on the current set S.
    Priority = - (number of conflicts it would create if added) + small random tie-breaker.
    Actually, to be efficient, we can't re-rank ALL vectors every step if n is large.
    But for small n (which is likely, given 300s limit and 3^n growth), we can.
    
    If n is large, 3^n is too big. 
    The task says "vectors of {0,1,2}^n".
    If n=10, 3^10 = 59049. 
    If n=12, 3^12 = 531441.
    If n=15, 3^15 = 14M. Too slow for dynamic re-ranking in Python.
    
    We need to know n. The function signature is priority(vector, n).
    The evaluator calls priority for each vector to sort them? 
    "vectors ... are added greedily in descending priority"
    This implies the evaluator sorts the vectors based on the priority returned by our function.
    It does NOT imply we can control the order dynamically.
    The interface is: `priority(vector: tuple, n: int) -> float`.
    The evaluator likely:
    1. Generates all vectors.
    2. Calls priority(v, n) for each v.
    3. Sorts them.
    4. Greedily adds them.
    
    This means my "dynamic" strategy is **impossible** with the given interface!
    The interface is static. I cannot see the set S when calculating the priority for a vector.
    
    RE-READING TASK:
    "Interface: priority(vector: tuple, n: int) -> float; vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
    
    This confirms the priority is calculated **once** per vector, independent of the set S.
    Therefore, I must stick to a **static** priority function, but I can make it much smarter than simple sums.
    
    What makes a good static priority for cap sets?
    Vectors with "balanced" coordinates (mix of 0, 1, 2) tend to be less likely to form APs with each other?
    Or vectors with specific weights?
    
    Let's look at the "inspiration" again.
    Inspiration 1 used static strategies.
    Inspiration 2 mentioned "dynamic greedy search" but the interface prevents it.
    Maybe Inspiration 2 simulated the greedy process internally to *choose* a good static heuristic?
    
    Yes! That's the key.
    I can use the available time (300s) to **simulate** the greedy process with various static heuristics, 
    find the one that yields the largest cap set for a given n, and then use that heuristic for the actual evaluation.
    
    BUT, the evaluator calls `priority` for the ACTUAL n.
    If n is large, I can't simulate the full greedy process for the actual n in 300s.
    However, I can simulate for a **smaller** n (e.g., n=5 or n=6) to find a good *type* of heuristic, 
    or I can use the time to optimize the *parameters* of the heuristic.
    
    Alternatively, if n is small (e.g., n <= 8), I can run the actual greedy simulation with several candidate heuristics 
    and pick the best one.
    
    Let's check the complexity.
    If n=8, 3^8 = 6561 vectors.
    Greedy check: For each vector, check against S. 
    Worst case S size ~ 3^4 = 81? No, max cap set size for n=8 is around 100-200?
    Actually, max cap set size in Z_3^8 is 255? No, that's for n=4?
    Cap set size grows as ~ 2.75^n.
    For n=8, size ~ 2.75^8 ~ 3000?
    Checking 6561 vectors against a set of 3000 vectors.
    6561 * 3000 = 20M operations. In Python, this might take 10-20 seconds.
    So for n=8, I can run ~10-20 simulations.
    For n=10, 3^10 = 59049. Size ~ 2.75^10 ~ 20000.
    59049 * 20000 = 1.2B operations. Too slow for Python in 300s.
    
    So, strategy:
    1. On first call, determine n.
    2. If n is small (<= 7 or 8), run a bootstrap search:
       - Define a family of parametric heuristics (e.g., weighted sums with weights w0, w1, w2).
       - Randomly sample parameters.
       - For each parameter set, run the greedy algorithm to find the cap set size.
       - Keep the best parameters.
    3. If n is large (> 8), use a pre-computed "best" heuristic found from small n, or a theoretically motivated one (like balanced coordinates).
    
    Let's implement this "Bootstrap Search for Heuristic Parameters" strategy.
    
    Heuristic Family:
    priority(v) = sum(w_i * v_i)
    where w_i are weights for coordinates 0, 1, 2.
    Or priority(v) = - sum(c_i * |v_i - 1|^2) ? (Favors 1s)
    Or priority(v) = - (sum(v) - n/2)^2 ? (Favors balanced sums)
    
    Let's try a weighted sum: P(v) = w0*count_0(v) + w1*count_1(v) + w2*count_2(v).
    We want to find w0, w1, w2 that maximize the resulting cap set size.
    
    To make it faster, we can use a simplified greedy check or just run the full one for small n.
    """
    global _start_time, _best_cap_size, _best_weights, _initialized
    
    if _start_time is None:
        _start_time = time.time()
        
    if not _initialized:
        # If n is small enough, do a bootstrap search
        if n <= 7:
            _bootstrap_search(n)
        else:
            # For large n, use a heuristic known to work well: 
            # Favor vectors with more 1s? Or balanced?
            # Literature suggests balanced vectors are good.
            # Let's use a default that favors "balanced" counts.
            _best_weights = (1.0, 1.0, 1.0) # All equal, so priority is constant? No.
            # If weights are equal, priority is sum(v).
            # Let's try to optimize for n=5 quickly and extrapolate?
            # Or just use a fixed good heuristic.
            # A common good heuristic is to minimize the variance of coordinates?
            # Let's stick to weighted sum. 
            # For large n, we can't search. We'll use weights that favor 1s slightly?
            # Or maybe random weights?
            # Let's use the best weights found for n=5 if we ran it? 
            # But we didn't run it if n>7.
            # Let's run a quick search for n=5 regardless, to get some weights, 
            # and hope they generalize.
            _bootstrap_search(5)
            
        _initialized = True

    # Return priority based on best weights
    w0, w1, w2 = _best_weights
    count_0 = sum(1 for x in vector if x == 0)
    count_1 = sum(1 for x in vector if x == 1)
    count_2 = sum(1 for x in vector if x == 2)
    return w0 * count_0 + w1 * count_1 + w2 * count_2

def _bootstrap_search(n):
    """
    Search for optimal weights (w0, w1, w2) for the priority function
    P(v) = w0*c0 + w1*c1 + w2*c2
    by running the greedy cap set algorithm for a fixed n.
    """
    global _best_weights, _start_time
    
    deadline = _start_time + 280.0  # Leave 20s margin
    
    vectors = list(product([0, 1, 2], repeat=n))
    
    # Pre-compute counts for each vector to speed up priority calculation
    counts = []
    for v in vectors:
        c0 = v.count(0)
        c1 = v.count(1)
        c2 = v.count(2)
        counts.append((c0, c1, c2))
        
    best_size = 0
    best_w = (1.0, 1.0, 1.0)
    
    # Try a few random weight combinations
    # Also try some structured ones
    candidates = []
    # Structured
    candidates.append((0, 1, 0)) # Favor 1s
    candidates.append((0, 0, 1)) # Favor 2s
    candidates.append((1, 0, 0)) # Favor 0s
    candidates.append((1, 1, 1)) # Sum
    candidates.append((1, 0, -1)) # Diff
    
    # Random
    import random
    for _ in range(50):
        w0 = random.uniform(-1, 1)
        w1 = random.uniform(-1, 1)
        w2 = random.uniform(-1, 1)
        candidates.append((w0, w1, w2))
        
    for w in candidates:
        if time.time() > deadline:
            break
            
        w0, w1, w2 = w
        # Calculate priorities
        prios = [w0*c0 + w1*c1 + w2*c2 for c0, c1, c2 in counts]
        
        # Sort indices by priority descending
        indices = sorted(range(len(vectors)), key=lambda i: -prios[i])
        
        S = set()
        # Optimization: Store vectors as tuples. 
        # To speed up AP check, we can use numpy? 
        # But for n<=7, Python sets are okay.
        
        for idx in indices:
            v = vectors[idx]
            v_arr = np.array(v)
            
            can_add = True
            for u in S:
                u_arr = np.array(u)
                w1_cand = (2 * v_arr - u_arr) % 3
                if tuple(w1_cand) in S:
                    can_add = False
                    break
                w2_cand = (2 * u_arr - v_arr) % 3
                if tuple(w2_cand) in S:
                    can_add = False
                    break
            
            if can_add:
                S.add(v)
                
        size = len(S)
        if size > best_size:
            best_size = size
            best_w = w
            # Print for debugging? No, just update.
            
    _best_weights = best_w

# Global variable for weights
_best_weights = (1.0, 1.0, 1.0)

def priority(vector, n):
    global _start_time, _initialized
    
    if _start_time is None:
        _start_time = time.time()
        
    if not _initialized:
        if n <= 7:
            _bootstrap_search(n)
        else:
            # For large n, search on n=5 to get a good heuristic
            _bootstrap_search(5)
        _initialized = True

    w0, w1, w2 = _best_weights
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    return w0 * c0 + w1 * c1 + w2 * c2
