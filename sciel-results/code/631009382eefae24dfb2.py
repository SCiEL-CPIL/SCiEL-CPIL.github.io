import time
import random
from itertools import combinations

N = 192
MAX_SIZE = 48
K = 3

def construct():
    start = time.time()
    deadline = start + 85.0 # Safety margin
    
    # Precompute all valid triples (a, b, c) from 1..N such that they are distinct.
    # There are ~1.1M triples. We can't store them all easily in a way that helps backtracking directly 
    # unless we index by sum or divisor.
    
    # Alternative Precomputation for Speed:
    # For a fixed set S, checking if x is safe requires:
    # 1. No (a,b,c) in S^3 such that x | (a+b+c).
    # 2. No a in S, b,c in S\{a} such that a | (x+b+c).
    
    # To make backtracking fast, we need very fast "is_safe" checks.
    # Let's precompute for every possible sum s (max sum 192*3 = 576), 
    # the list of divisors d such that d | s.
    # divisors_of_sum[s] = [d1, d2, ...]
    
    max_sum = 3 * N
    divisors_of_sum = [[] for _ in range(max_sum + 1)]
    for s in range(1, max_sum + 1):
        for d in range(1, N + 1):
            if s % d == 0:
                divisors_of_sum[s].append(d)
                
    # Precompute pair sums for the entire domain 1..N?
    # When adding x to S, we need to check if any a in S divides x+b+c.
    # This is equivalent to: does there exist b,c in S\{a} such that (x+b+c) % a == 0?
    
    # Strategy: 
    # 1. Start with a strong greedy seed (Large numbers first, as they are less likely to divide sums).
    # 2. Use Backtracking to try to reach MAX_SIZE.
    #    - Maintain S (list) and S_set (set).
    #    - Maintain "Conflicts" or just re-check.
    #    - Heuristic: Try to add numbers. If stuck, backtrack.
    #    - Variable Ordering: Try to add the number that has the "fewest conflicts" or is "most constrained"?
    #      Actually, for maximization, we want to add as many as possible.
    #      MRV (Minimum Remaining Values) usually applies to assigning values to variables.
    #      Here, variables are "slots" in the set? Or we just pick candidates.
    
    # Let's use a recursive backtracking with a "candidate pool".
    # To avoid O(N^3) checks in Python loops which are slow, we can optimize the check.
    
    def is_safe_add(x, S, S_set):
        n = len(S)
        
        # Check 1: x divides sum of any 3 in S
        if n >= 3:
            # Optimization: 
            # We need to know if any a+b+c is a multiple of x.
            # Instead of iterating all triples, we can use the precomputed divisors?
            # No, divisors_of_sum is sum -> divisors.
            # We need: for any triple in S, is x in divisors_of_sum[sum]?
            # This is still O(n^3).
            
            # Alternative: 
            # Precompute for each x, a set of "forbidden pair sums" mod x?
            # If we have S, we can compute pair_sums = {a+b for a,b in S, a<b}.
            # Then for each c in S, check if (x-c) % x is in pair_sums?
            # Wait, a+b+c = k*x => a+b = k*x - c.
            # So (a+b) % x == (-c) % x.
            
            # Let's implement this O(n^2) check.
            # 1. Compute set of pair sums mod x for distinct a,b in S.
            #    But we need a,b,c distinct.
            #    If a+b+c is a multiple of x, and a,b,c are distinct.
            #    Let P_x = { (a+b) % x | a,b in S, a != b }
            #    For each c in S, if (-c) % x in P_x, we need to ensure the pair (a,b) that produced that sum does not include c.
            #    This is tricky if there are multiple pairs with the same sum mod x.
            
            # Given n <= 48, O(n^3) is 48^3 = 110k ops.
            # In Python, 110k simple ops is ~1-2ms.
            # Backtracking depth 48. If we branch factor 10, that's 10^10 checks. Too slow.
            # We need pruning.
            
            # Let's stick to the O(n^3) but with early exit and C-optimized itertools if possible?
            # Actually, let's try to optimize the check using NumPy?
            # No, standard library is safer for logic.
            
            # Let's use the residue map method which is O(n^2 + n*unique_residues).
            # For Check 1:
            # We want to know if there exist distinct a,b,c in S such that a+b+c ≡ 0 (mod x).
            
            # If x is large (e.g., > N/2), then a+b+c < 3N.
            # Multiples of x in range [3, 3N] are x, 2x, 3x...
            # For x > N/2, only x, 2x, 3x are possible?
            # Max sum 576. Min sum 6.
            # If x=192, multiples: 192, 384, 576.
            
            # Let's just do the O(n^3) check but break early.
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # Check 2: a in S divides x+b+c
        if n >= 2:
            for a in S:
                target = (-x) % a
                # We need b, c in S\{a} distinct such that (b+c) % a == target.
                # Let's build a map of residues of S\{a}.
                res_count = {}
                for v in S:
                    if v == a: continue
                    r = v % a
                    res_count[r] = res_count.get(r, 0) + 1
                
                # Now iterate b in S\{a}
                # We need c such that c % a == (target - b%a) % a
                # And c != b.
                
                # To avoid O(n^2) inner loop, we can just check existence.
                # If we find any b, we check if there is a valid c.
                # Since we just need existence, we can iterate b and check the count.
                
                found = False
                # Optimization: Iterate over unique residues in S\{a}?
                # No, we need specific b to check c != b.
                # But if res_count[rc_needed] > 0, does there exist c != b?
                # If rc_needed != rb, then any c with residue rc_needed is distinct from b (since values are distinct? No, residues distinct implies values distinct? 
                # Wait. If r1 != r2, v1 != v2. Yes.
                # So if rc_needed != rb, and res_count[rc_needed] > 0, we have a valid pair.
                # If rc_needed == rb, we need res_count[rb] >= 2 (one is b, need another).
                
                for b in S:
                    if b == a: continue
                    rb = b % a
                    rc_needed = (target - rb) % a
                    if rc_needed == rb:
                        if res_count.get(rc_needed, 0) >= 2:
                            found = True
                            break
                    else:
                        if res_count.get(rc_needed, 0) >= 1:
                            found = True
                            break
                if found:
                    return False
        return True

    # To make backtracking viable, we need a good initial guess and strong pruning.
    # Greedy "Largest First" is a good start.
    
    # Let's run a Greedy to get a base set.
    S = []
    S_set = set()
    pool = list(range(N, 0, -1))
    
    for x in pool:
        if time.time() > deadline - 5: break
        if len(S) >= MAX_SIZE: break
        if is_safe_add(x, S, S_set):
            S.append(x)
            S_set.add(x)
            
    best_size = len(S)
    best_S = S[:]
    
    # Now, try to improve using Backtracking / Local Search.
    # Since full backtracking is likely too slow, we will use a "Greedy Repair" or "Simulated Annealing" style local search 
    # but with a focus on "Swaps" that increase size or maintain size while allowing future adds.
    
    # Actually, let's try a specific CSP approach:
    # We know we want 48 elements.
    # Let's fix the "hard" constraints. Large numbers are safe.
    # Small numbers are dangerous.
    
    # Strategy: 
    # 1. Start with the Greedy solution.
    # 2. If size < 48, identify "blocked" candidates.
    # 3. Try to remove one element from S to unblock two candidates.
    
    def local_search_improve(S, S_set, deadline):
        current_S = S[:]
        current_set = set(S)
        current_size = len(current_S)
        max_iter = 1000
        no_improve_count = 0
        
        while current_size < MAX_SIZE and no_improve_count < 50 and time.time() < deadline - 2:
            improved = False
            remaining = [x for x in range(1, N+1) if x not in current_set]
            
            # Try to add any remaining element
            for x in remaining:
                if is_safe_add(x, current_S, current_set):
                    current_S.append(x)
                    current_set.add(x)
                    current_size += 1
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                # Try 1-remove, 2-add
                # Remove x from current_S.
                # Try to add y, then z.
                # To speed up, only try removing small elements (most restrictive).
                candidates_to_remove = sorted(current_S)[:10] # Smallest 10
                
                for x_rm in candidates_to_remove:
                    if time.time() > deadline - 2: break
                    
                    temp_S = [v for v in current_S if v != x_rm]
                    temp_set = set(temp_S)
                    
                    # Try to add y
                    # We want to add y that is "safe" with temp_S
                    # And then z that is safe with temp_S + {y}
                    
                    # Sort remaining by "potential" (larger is better?)
                    # Let's just iterate remaining.
                    for y in remaining:
                        if y == x_rm: continue
                        if is_safe_add(y, temp_S, temp_set):
                            # Try to add z
                            temp_S2 = temp_S + [y]
                            temp_set2 = temp_set | {y}
                            
                            for z in remaining:
                                if z == y or z == x_rm: continue
                                if is_safe_add(z, temp_S2, temp_set2):
                                    # Success! We removed x_rm, added y, z.
                                    # New size: current_size - 1 + 2 = current_size + 1
                                    current_S = temp_S2 + [z]
                                    current_set = temp_set2 | {z}
                                    current_size += 1
                                    improved = True
                                    no_improve_count = 0
                                    break
                            if improved:
                                break
                    if improved:
                        break
                            
            if not improved:
                no_improve_count += 1
                # Random perturbation?
                # Swap a random element in S with a random element not in S if it keeps size same?
                # This might help escape local minima.
                if current_S and remaining:
                    idx_rm = random.randint(0, len(current_S)-1)
                    x_rm = current_S[idx_rm]
                    y_add = random.choice(remaining)
                    
                    temp_S = [v for i, v in enumerate(current_S) if i != idx_rm]
                    temp_set = set(temp_S)
                    
                    if is_safe_add(y_add, temp_S, temp_set):
                        current_S = temp_S + [y_add]
                        current_set = temp_set | {y_add}
                        # Size remains same, but structure changed.
                        # Reset no_improve_count slightly?
                        # no_improve_count = max(0, no_improve_count - 5)
                        
        return current_S, current_set

    # Run local search
    S, S_set = local_search_improve(best_S, set(best_S), deadline)
    
    # If we still haven't reached 48, try a few random restarts with different initial seeds?
    # Given time constraints, one good local search is better than many bad ones.
    # But let's do 3 restarts if time permits.
    
    if len(S) < MAX_SIZE:
        for _ in range(3):
            if time.time() > deadline - 5: break
            
            # Random initial greedy
            S_init = []
            S_init_set = set()
            pool = list(range(1, N+1))
            random.shuffle(pool)
            # Mix of large and random
            for x in pool:
                if len(S_init) >= MAX_SIZE: break
                if is_safe_add(x, S_init, S_init_set):
                    S_init.append(x)
                    S_init_set.add(x)
            
            S_new, S_new_set = local_search_improve(S_init, S_init_set, deadline)
            
            if len(S_new) > len(S):
                S = S_new
                S_set = S_new_set
                if len(S) == MAX_SIZE:
                    break
                    
    return S

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(res)
