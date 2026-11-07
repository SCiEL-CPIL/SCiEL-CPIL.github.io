import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute comparability matrix: comp[a][b] = 1 if a and b are comparable
    # Use a flat array for faster access: comp_flat[a * N + b]
    comp_flat = bytearray(N * N)
    for a in range(N):
        for b in range(N):
            if (a & b) == a or (a & b) == b:
                comp_flat[a * N + b] = 1
                comp_flat[b * N + a] = 1  # symmetric
    
    def is_comp(a, b):
        return comp_flat[a * N + b]
    
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_comp(family[i], family[j]):
                    count += 1
        return count
    
    def local_search(family, max_iters, deadline):
        # family is a list of masks
        # comp_in_F[y] = number of sets in family comparable to y
        comp_in_F = [0] * N
        
        # Efficient initialization using precomputed matrix
        for y in range(N):
            c = 0
            for z in family:
                if comp_flat[y * N + z]:
                    c += 1
            comp_in_F[y] = c
            
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        
        # We store family as a list of masks
        # We also maintain a set of masks in family for quick lookup
        fam_set = set(family)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try to improve each position
            for i in range(m):
                x = family[i]
                # Number of other sets in family comparable to x
                k_x = comp_in_F[x] - 1
                
                # Find best y not in family
                # Maximize V(y) = comp_in_F[y] - (1 if y~x else 0)
                best_y = -1
                best_val = -1
                
                # Optimization: iterate through all y in range(N)
                # This is 4096 iterations.
                for y in range(N):
                    if y in fam_set:
                        continue
                    val = comp_in_F[y]
                    if comp_flat[y * N + x]:
                        val -= 1
                    
                    if val > best_val:
                        best_val = val
                        best_y = y
                
                if best_y == -1:
                    continue
                
                # Check if improvement
                # New contribution of best_y to the rest of the family (excluding x)
                # If we replace x with best_y:
                # Score change = (contrib of best_y to F\{x}) - (contrib of x to F\{x})
                # contrib of x to F\{x} is k_x.
                # contrib of best_y to F\{x} is best_val (calculated as comp_in_F[best_y] minus 1 if comparable to x)
                
                if best_val > k_x:
                    # Perform swap
                    # Update comp_in_F
                    # For all z in range(N):
                    # comp_in_F[z] += (1 if z~best_y else 0) - (1 if z~x else 0)
                    
                    for z in range(N):
                        delta = 0
                        if comp_flat[z * N + best_y]:
                            delta += 1
                        if comp_flat[z * N + x]:
                            delta -= 1
                        if delta != 0:
                            comp_in_F[z] += delta
                    
                    # Update family and set
                    fam_set.remove(x)
                    fam_set.add(best_y)
                    family[i] = best_y
                    current_score += best_val - k_x
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
                        
                    improved = True
            
            if not improved:
                break
                
        return best_family, best_score

    best_overall = None
    best_score_overall = 0
    
    # Multiple restarts
    # Use a few different random seeds
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        random.seed(42 + restart)
        # Random initialization
        family = random.sample(range(N), m)
        
        fam, score = local_search(family, 10000, deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam
    
    # One more try with a chain-based init
    if time.time() < deadline - 15:
        chain = [0]
        cur = 0
        for b in range(n):
            cur |= (1 << b)
            chain.append(cur)
        remaining = [x for x in range(N) if x not in chain]
        random.seed(999)
        extra = random.sample(remaining, m - len(chain))
        family = chain + extra
        fam, score = local_search(family, 10000, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam

    if best_overall is None:
        best_overall = list(range(m))
    
    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
