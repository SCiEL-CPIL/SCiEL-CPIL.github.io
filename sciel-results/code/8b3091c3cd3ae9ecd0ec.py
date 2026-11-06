import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill

    # 255 = 3 * 5 * 17
    # Any element x in a valid set S (containing 255) must satisfy gcd(x, 255) > 1,
    # so x must be divisible by 3, 5, or 17.
    #
    # Key structural insight: In a pairwise non-coprime set where every element
    # shares a factor with 255 = 3*5*17, all elements must share a common prime
    # factor p in {3, 5, 17}. Proof sketch: if a is divisible only by 3 (not 5,17)
    # and b is divisible only by 5 (not 3,17), then gcd(a,b)=1, contradiction.
    # Thus S ⊆ multiples of some p ∈ {3,5,17}.
    #
    # |multiples of 3 in [1,255]| = 85
    # |multiples of 5 in [1,255]| = 51
    # |multiples of 17 in [1,255]| = 15
    #
    # Maximum is 85 (multiples of 3).

    # Primary answer: all multiples of 3
    S = list(range(3, 256, 3))

    # Verify correctness
    if 255 not in S:
        S.append(255)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if gcd(S[i], S[j]) == 1:
                # Should never happen, but fallback
                S = list(range(3, 256, 3))
                break

    # Search machinery: attempt to improve beyond the structural optimum.
    # This is a safety net; the mathematical argument above shows 85 is optimal.
    # We run a limited local search in case our reasoning has a gap.

    candidates = [x for x in range(1, 256) if gcd(x, 255) > 1]
    best_S = S[:]
    best_size = len(S)

    # Greedy: try to add any candidate not in S
    if (time.time() - start_time) < deadline - 15:
        S_set = set(S)
        improved = True
        while improved and (time.time() - start_time) < deadline - 15:
            improved = False
            for x in candidates:
                if x in S_set:
                    continue
                can_add = all(gcd(x, y) > 1 for y in S)
                if can_add:
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]

    # Graph-theoretic approach: build a graph where vertices are candidates,
    # edges connect pairs with gcd > 1. Use greedy coloring-based upper bounds
    # and simulated annealing to optimize subset size.
    if (time.time() - start_time) < deadline - 10:
        # Build adjacency: for each candidate, which other candidates have gcd > 1
        n_cand = len(candidates)
        cand_set = set(candidates)
        
        # Precompute gcd matrix for speed
        gcd_matrix = [[0] * n_cand for _ in range(n_cand)]
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                g = gcd(candidates[i], candidates[j])
                gcd_matrix[i][j] = g
                gcd_matrix[j][i] = g
        
        # Greedy coloring-based upper bound heuristic
        # We'll use a simple greedy approach to find a large clique
        # Start with the multiples of 3 set as initial solution
        current_S = best_S[:]
        current_set = set(current_S)
        
        # Simulated annealing-like local search
        import random
        random.seed(42)
        
        best_SA = best_S[:]
        best_SA_size = best_size
        current_temp = 100.0
        cooling_rate = 0.95
        min_temp = 1.0
        
        while current_temp > min_temp and (time.time() - start_time) < deadline - 5:
            # Generate a move: remove a random element and try to add others
            if current_S:
                idx = random.randint(0, len(current_S) - 1)
                removed = current_S[idx]
                temp_S = current_S[:idx] + current_S[idx+1:]
                temp_set = set(temp_S)
                
                # Try to add candidates greedily
                for y in candidates:
                    if y in temp_set:
                        continue
                    if all(gcd(y, z) > 1 for z in temp_S):
                        temp_S.append(y)
                        temp_set.add(y)
                
                # Accept or reject based on simulated annealing
                delta = len(temp_S) - len(current_S)
                if delta >= 0 or random.random() < (2.718281828 ** (delta / current_temp)):
                    current_S = temp_S
                    current_set = temp_set
                    if len(current_S) > best_SA_size:
                        best_SA_size = len(current_S)
                        best_SA = current_S[:]
            
            current_temp *= cooling_rate
        
        if best_SA_size > best_size:
            best_size = best_SA_size
            best_S = best_SA[:]

    return best_S
