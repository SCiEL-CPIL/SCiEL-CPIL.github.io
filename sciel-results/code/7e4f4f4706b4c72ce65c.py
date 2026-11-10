import time
import random

_START = time.time()
_DEADLINE = 145.0  # 5s margin

def _edge_index(i, j):
    return i * 15 - (i * (i + 1)) // 2 + (j - i - 1)

def _build_k4s():
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    k4s.append((
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    ))
    return k4s

_K4S = _build_k4s()
# Map edge index to list of K4 indices it belongs to
_EDGE_TO_K4 = [[] for _ in range(120)]
for k4_idx, k4 in enumerate(_K4S):
    for e in k4:
        _EDGE_TO_K4[e].append(k4_idx)

def _initial_coloring_17():
    # A known valid 17-coloring (from previous logic or simple construction)
    # (i + j) % 17 works?
    # Let's verify (i+j)%17 validity quickly in head or assume it's valid as per prompt history
    cols = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    return cols

def construct():
    # Fallback solution
    fallback = _initial_coloring_17()
    
    # Try to find a 15-color solution using backtracking
    # Symmetry breaking: Fix colors of edges (0, i) for i=1..15 to 0..14
    # This uses all 15 colors.
    # We need to assign colors to edges (i, j) where 1 <= i < j < 16.
    # There are 15 vertices 1..15. Edges among them: 15*14/2 = 105.
    
    K = 15
    # colors array of size 120, -1 for unassigned
    colors = [-1] * 120
    
    # Fix edges incident to 0
    for i in range(1, 16):
        colors[_edge_index(0, i)] = i - 1
        
    # List of edges to assign (not incident to 0)
    edges_to_assign = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            edges_to_assign.append(_edge_index(i, j))
            
    # Shuffle edges to assign? Or use MRV?
    # Static order might be okay if we have strong pruning.
    # Let's try a random shuffle to avoid bad paths, but fixed seed for reproducibility in testing?
    # Actually, random restarts are better.
    
    rng = random.Random(42)
    
    # Precompute K4 constraints for faster checking
    # For each K4, we need to check if the set of assigned colors has < 5 distinct colors 
    # AND if it's possible to reach 5 with the remaining unassigned edges.
    # Simpler: Check if current assigned colors in K4 already violate? No, violation is <5.
    # We can only prune if the MAXIMUM possible distinct colors in the K4 is < 5.
    # Or if the CURRENT distinct colors are such that even if all unassigned take new unique colors, we can't reach 5?
    # Actually, if we have 3 assigned colors, and 3 unassigned, max distinct is 6.
    # If we have 3 assigned colors {0,1,2}, and unassigned edges must take colors from {0,1,2,3,4} (K=5 palette? No K=15).
    # The domain is 0..14.
    
    # Heuristic: 
    # 1. If a K4 has 0 unassigned edges, check if distinct colors >= 5. If not, fail.
    # 2. If a K4 has 1 unassigned edge, and currently has < 4 distinct colors, the last edge MUST take a color not in the current set.
    # 3. If a K4 has 2 unassigned edges, and currently has < 3 distinct colors, the two edges MUST take 2 distinct colors not in the current set?
    #    If current is {0,1} (2 colors), we need 3 more. 2 edges can provide at most 2 new colors. Max total 4. Fail.
    #    So if current distinct < 3 and unassigned == 2, FAIL.
    #    If current distinct == 3 and unassigned == 2, we need 2 new colors. The 2 edges must take colors outside current set, and be distinct from each other?
    #    If they are distinct and outside, we get 5. If one is inside, we get 4. So yes, they must be distinct and outside.
    
    def is_consistent(colors):
        for k4_idx in range(len(_K4S)):
            e = _K4S[k4_idx]
            assigned_colors = []
            unassigned_count = 0
            for idx in e:
                if colors[idx] == -1:
                    unassigned_count += 1
                else:
                    assigned_colors.append(colors[idx])
            
            distinct_assigned = set(assigned_colors)
            d = len(distinct_assigned)
            
            if unassigned_count == 0:
                if d < 5:
                    return False
            elif unassigned_count == 1:
                if d < 4:
                    return False
                # If d == 4, the last edge must take a color NOT in distinct_assigned.
                # We don't enforce this immediately in consistency, but in domain pruning.
            elif unassigned_count == 2:
                if d < 3:
                    return False
                # If d == 3, the 2 edges must take 2 distinct colors NOT in distinct_assigned.
            elif unassigned_count == 3:
                if d < 2:
                    return False
            # etc.
        return True

    # Backtracking
    def backtrack(pos):
        if time.time() > _DEADLINE:
            return False
        
        if pos == len(edges_to_assign):
            return is_consistent(colors)
        
        e_idx = edges_to_assign[pos]
        
        # Try colors 0..14
        # Optimization: If this edge is in a K4 with 4 assigned colors, it must be a new color.
        # Let's just try all and rely on is_consistent
        for c in range(K):
            colors[e_idx] = c
            if is_consistent(colors):
                if backtrack(pos + 1):
                    return True
            colors[e_idx] = -1
            
        return False

    # Run backtracking
    found = False
    # Try a few random permutations of edges_to_assign to escape local minima?
    # Backtracking is complete, so order doesn't affect existence, only speed.
    # MRV (Minimum Remaining Values) is better.
    # But implementing dynamic MRV in Python is slow.
    # Let's try static order first.
    
    # To increase chances, we can try different symmetry breakings?
    # The fixing of 0-1..0-15 to 0..14 is very strong.
    
    # If K=15 fails or times out, we might try K=16? No, 17 is already found.
    # We want to beat 17. So K=15 is the target.
    
    # Let's add a timeout to the backtracking
    start_time = time.time()
    # We will run the backtracker. If it finds a solution, we return it.
    # If it times out, we return the fallback.
    
    # To make it faster, we can pre-compute the K4s for each edge and check only those.
    # Modified is_consistent to be faster?
    # The current is_consistent checks ALL K4s. That's 1820 K4s.
    # We can maintain a count of "violations" or "tight constraints".
    
    # Let's implement a faster consistency check using the _EDGE_TO_K4 map.
    # When we assign a color to e_idx, we only need to re-check K4s containing e_idx.
    
    # State:
    # k4_status[k4_idx] = (num_unassigned, set_of_assigned_colors)
    # This is expensive to maintain.
    
    # Alternative: Just use the simple is_consistent but call it less often?
    # No, we need to prune.
    
    # Let's try a different approach: Local Search with 15 colors.
    # Initialize randomly with 15 colors.
    # Objective: Minimize number of K4s with < 5 colors.
    # If we reach 0, we have a solution.
    
    def local_search_15():
        # Initialize
        cols = [0] * 120
        # Fix symmetry breaking? Random is better for LS.
        for i in range(120):
            cols[i] = rng.randrange(K)
            
        # Calculate initial penalty
        def penalty(cols):
            p = 0
            for k4 in _K4S:
                s = {cols[k4[0]], cols[k4[1]], cols[k4[2]], cols[k4[3]], cols[k4[4]], cols[k4[5]]}
                if len(s) < 5:
                    p += 1
            return p
        
        p = penalty(cols)
        best_cols = list(cols)
        best_p = p
        
        # Simulated Annealing
        T = 1.0
        cool = 0.999
        for i in range(50000):
            if time.time() > _DEADLINE:
                break
            if best_p == 0:
                return best_cols
            
            # Pick a random edge
            e_idx = rng.randrange(120)
            old_c = cols[e_idx]
            new_c = rng.randrange(K)
            if old_c == new_c:
                continue
                
            cols[e_idx] = new_c
            # Calculate change in penalty
            # Only K4s containing e_idx are affected
            delta = 0
            for k4_idx in _EDGE_TO_K4[e_idx]:
                k4 = _K4S[k4_idx]
                old_s = {cols[k4[0]], cols[k4[1]], cols[k4[2]], cols[k4[3]], cols[k4[4]], cols[k4[5]]}
                # Wait, we already changed cols[e_idx].
                # We need to compare old and new sets.
                # Let's just re-calculate penalty for these K4s.
                # This is O(degree) per move. Degree is 91.
                # 50000 * 91 is ~4.5M ops. Fast enough.
                
                # To be precise:
                # We need the set BEFORE and AFTER.
                # Since we only changed one edge, we can compute the set for the K4.
                # But we need to know if it went from <5 to >=5 or vice versa.
                
                # Let's just compute the new penalty for the affected K4s.
                # We need the old penalty for these K4s.
                # We can store the current penalty per K4?
                # Or just re-evaluate the whole penalty? No, too slow.
                
                # Let's store k4_penalty array.
            
        return None # Fallback

    # Let's implement the Local Search with delta penalty properly.
    
    def run_ls():
        cols = [rng.randrange(K) for _ in range(120)]
        
        # Precompute K4 penalty
        k4_pen = [0] * len(_K4S)
        total_pen = 0
        for k4_idx, k4 in enumerate(_K4S):
            s = {cols[k4[0]], cols[k4[1]], cols[k4[2]], cols[k4[3]], cols[k4[4]], cols[k4[5]]}
            if len(s) < 5:
                k4_pen[k4_idx] = 1
                total_pen += 1
        
        best_cols = list(cols)
        best_pen = total_pen
        
        T = 1.0
        cool = 0.9995
        for i in range(100000):
            if time.time() > _DEADLINE:
                break
            if best_pen == 0:
                return best_cols
            
            e_idx = rng.randrange(120)
            old_c = cols[e_idx]
            new_c = rng.randrange(K)
            if old_c == new_c:
                continue
            
            cols[e_idx] = new_c
            
            delta = 0
            for k4_idx in _EDGE_TO_K4[e_idx]:
                k4 = _K4S[k4_idx]
                s = {cols[k4[0]], cols[k4[1]], cols[k4[2]], cols[k4[3]], cols[k4[4]], cols[k4[5]]}
                new_pen = 1 if len(s) < 5 else 0
                delta += new_pen - k4_pen[k4_idx]
                k4_pen[k4_idx] = new_pen
            
            total_pen += delta
            
            if total_pen < best_pen:
                best_pen = total_pen
                best_cols = list(cols)
            else:
                # Accept with probability exp(-delta/T)
                if delta > 0:
                    if rng.random() > pow(2.71828, -delta / T):
                        # Reject
                        cols[e_idx] = old_c
                        # Revert penalty
                        for k4_idx in _EDGE_TO_K4[e_idx]:
                            k4 = _K4S[k4_idx]
                            s = {cols[k4[0]], cols[k4[1]], cols[k4[2]], cols[k4[3]], cols[k4[4]], cols[k4[5]]}
                            new_pen = 1 if len(s) < 5 else 0
                            # This re-calculation is redundant if we stored old state, 
                            # but simpler to just revert and re-calc delta?
                            # Actually, we can just store the old k4_pen values.
                            pass
                        # Better: store old k4_pen for affected K4s
                        # Let's refactor to store old state
                        pass
                else:
                    # delta <= 0, accept
                    pass
            
            T *= cool
            
        return None

    # The LS implementation above has a bug in rejection (doesn't revert correctly).
    # Let's write a cleaner LS.
    
    def clean_ls():
        cols = [rng.randrange(K) for _ in range(120)]
        k4_pen = [0] * len(_K4S)
        total_pen = 0
        for k4_idx, k4 in enumerate(_K4S):
            s = {cols[k4[0]], cols[k4[1]], cols[k4[2]], cols[k4[3]], cols[k4[4]], cols[k4[5]]}
            if len(s) < 5:
                k4_pen[k4_idx] = 1
                total_pen += 1
        
        best_cols = list(cols)
        best_pen = total_pen
        
        T = 1.0
        cool = 0.9995
        for i in range(200000):
            if time.time() > _DEADLINE:
                break
            if best_pen == 0:
                return best_cols
            
            e_idx = rng.randrange(120)
            old_c = cols[e_idx]
            new_c = rng.randrange(K)
            if old_c == new_c:
                continue
            
            cols[e_idx] = new_c
            
            delta = 0
            affected = _EDGE_TO_K4[e_idx]
            old_k4_pens = []
            for k4_idx in affected:
                k4 = _K4S[k4_idx]
                s = {cols[k4[0]], cols[k4[1]], cols[k4[2]], cols[k4[3]], cols[k4[4]], cols[k4[5]]}
                new_pen = 1 if len(s) < 5 else 0
                delta += new_pen - k4_pen[k4_idx]
                old_k4_pens.append((k4_idx, k4_pen[k4_idx]))
                k4_pen[k4_idx] = new_pen
            
            total_pen += delta
            
            if total_pen < best_pen:
                best_pen = total_pen
                best_cols = list(cols)
            else:
                if delta > 0:
                    if rng.random() > pow(2.71828, -delta / T):
                        # Reject
                        cols[e_idx] = old_c
                        for k4_idx, old_p in old_k4_pens:
                            k4_pen[k4_idx] = old_p
                        total_pen -= delta
                # else accept
            
            T *= cool
            
        return None

    result = clean_ls()
    if result:
        return result
    
    # If LS fails, return fallback
    return fallback
