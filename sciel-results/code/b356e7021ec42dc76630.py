import time
from math import log, floor, sqrt
import random
import numpy as np
from itertools import combinations

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Faster scoring using numpy for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all sums (i <= j)
    i_idx = np.repeat(np.arange(n), n)
    j_idx = np.tile(np.arange(n), n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    sums_unique = np.unique(sums)
    
    # Compute all differences (i < j), then add negatives and 0
    i_idx2 = np.repeat(np.arange(n), n-1)
    j_idx2 = []
    for i in range(n):
        j_idx2.extend(range(i+1, n))
    j_idx2 = np.array(j_idx2, dtype=np.int64)
    diffs_pos = U_arr[j_idx2] - U_arr[i_idx2]
    diffs = np.concatenate([diffs_pos, -diffs_pos, [0]])
    diffs_unique = np.unique(diffs)
    
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def modular_scaled_set(V, prime):
    """Scale set V by a prime to eliminate low-order sum collisions."""
    return sorted([v * prime for v in V])

def greedy_optimize_V(V_seed, prime, max_elements=30, deadline=None):
    """
    Greedy optimization of subset V to maximize difference/sum ratio.
    The actual set U = {v * prime for v in V}.
    """
    V = sorted(set(V_seed))
    if 0 not in V:
        V.insert(0, 0)
    
    current_score = score_set(modular_scaled_set(V, prime))
    best_V = list(V)
    best_score = current_score
    
    # Generate candidate elements for V
    max_v = max(V) if V else 1
    candidates = set()
    for v in V:
        for delta in range(1, max_v + 2):
            candidates.add(v + delta)
            if v - delta > 0:
                candidates.add(v - delta)
    
    # Add some random candidates
    for _ in range(50):
        candidates.add(random.randint(1, max_v * 2))
    
    candidates = sorted([c for c in candidates if c > 0])
    
    no_improvement = 0
    max_no_improvement = 20
    
    while time.time() < (deadline - 2 if deadline else 100):
        if no_improvement > max_no_improvement:
            break
        
        improved = False
        
        # Try adding a candidate
        if len(V) < max_elements:
            random.shuffle(candidates)
            for c in candidates[:20]:
                if c in V:
                    continue
                new_V = sorted(V + [c])
                new_score = score_set(modular_scaled_set(new_V, prime))
                if new_score > current_score:
                    V = new_V
                    current_score = new_score
                    improved = True
                    # Regenerate candidates
                    max_v = max(V)
                    candidates = set()
                    for v in V:
                        for delta in range(1, max_v + 1):
                            candidates.add(v + delta)
                            if v - delta > 0:
                                candidates.add(v - delta)
                    for _ in range(30):
                        candidates.add(random.randint(1, max_v * 2))
                    candidates = sorted([c for c in candidates if c > 0 and c not in V])
                    break
        
        # Try removing an element
        if not improved and len(V) > 3:
            removable = [v for v in V if v != 0]
            random.shuffle(removable)
            for c in removable[:10]:
                new_V = [v for v in V if v != c]
                new_score = score_set(modular_scaled_set(new_V, prime))
                if new_score > current_score:
                    V = new_V
                    current_score = new_score
                    improved = True
                    break
        
        # Try replacing an element
        if not improved and len(V) > 1 and candidates:
            idx = random.randint(1, len(V) - 1)
            old_val = V[idx]
            random.shuffle(candidates)
            for new_val in candidates[:10]:
                if new_val == old_val or new_val in V:
                    continue
                new_V = V.copy()
                new_V[idx] = new_val
                new_V = sorted(set(new_V))
                if len(new_V) != len(V):
                    continue
                new_score = score_set(modular_scaled_set(new_V, prime))
                if new_score > current_score:
                    V = new_V
                    current_score = new_score
                    improved = True
                    break
        
        if improved:
            no_improvement = 0
            if current_score > best_score:
                best_score = current_score
                best_V = list(V)
        else:
            no_improvement += 1
    
    return best_score, best_V

def base_expansion_set(B, k, max_elements=None):
    """
    Generate a set using base-B expansion hierarchy.
    Elements are u = sum_{j=0}^{k-1} b_j * B^j where b_j in {0, 1}.
    This structurally limits sumset collisions to local digit-wise interactions.
    """
    if max_elements is None:
        max_elements = 2 ** k
    
    U = set()
    U.add(0)
    
    for mask in range(1, min(2 ** k, max_elements)):
        val = 0
        for j in range(k):
            if mask & (1 << j):
                val += B ** j
        U.add(val)
    
    return sorted(U)

def base_expansion_mixed(B, k, sparsity=0.5, max_elements=None):
    """
    Generate a set using base-B expansion with sparsity control.
    Only include elements with a certain fraction of active digits.
    """
    if max_elements is None:
        max_elements = 2 ** k
    
    U = set()
    U.add(0)
    
    for mask in range(1, min(2 ** k, max_elements)):
        # Count active digits
        active = bin(mask).count('1')
        # Include based on sparsity (include if active fraction is near sparsity)
        if abs(active / k - sparsity) <= 0.15 or active == k:
            val = 0
            for j in range(k):
                if mask & (1 << j):
                    val += B ** j
            U.add(val)
    
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:10]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Modular Arithmetic Strategy (NEW - from directive)
    # Try various primes and optimize V using greedy
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for prime in primes:
        if time.time() > deadline - 30:
            break
        
        # Try different seed sets for V
        v_seeds = [
            [0, 1, 3, 7, 12, 18],
            [0, 1, 4, 9, 16, 25],
            [0, 1, 3, 6, 10, 15],
            [0, 1, 2, 5, 9, 14],
            [0, 1, 4, 8, 13, 19],
        ]
        
        for v_seed in v_seeds:
            if time.time() > deadline - 25:
                break
            
            # Greedy optimize V
            v_score, v_optimized = greedy_optimize_V(v_seed, prime, max_elements=25, deadline=deadline)
            
            if v_score > best_score:
                # Convert V back to U by scaling with prime
                best_set = modular_scaled_set(v_optimized, prime)
                best_score = v_score

    # 3. Base-Expansion Hierarchy Search
    for B in [3, 4, 5, 6, 8, 10, 12, 16]:
        if time.time() > deadline - 20:
            break
        for k in range(2, 7):
            # Full base expansion
            U = base_expansion_set(B, k, max_elements=150)
            if len(U) >= 2:
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
            
            # Sparse base expansion
            for sparsity in [0.3, 0.5, 0.7]:
                if time.time() > deadline - 18:
                    break
                U = base_expansion_mixed(B, k, sparsity=sparsity, max_elements=150)
                if len(U) >= 2:
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)

    # 4. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 61)]
    for b in b_candidates:
        if time.time() > deadline - 15:
            break
        for m in range(2, 15):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    # 5. Dynamic Local Search on best set
    if time.time() < deadline - 5:
        local_score, local_set = local_search_dynamic(best_set, deadline, max_val_limit=1000, max_n_limit=35)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # 6. Final Polish: Fine-grained local search on the best set
    if time.time() < deadline - 1:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 0.5 and iteration < 200:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 0.5:
                    break
                old_val = current[i]
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element near the max
            if not improved and n < 20 and time.time() < deadline - 0.5:
                max_val = max(current)
                for new_val in range(max_val - 5, max_val + 6):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 0.5:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

# Ensure 0 is in the set and it's a list of nonnegative integers
# The function construct returns a list.
# The evaluator calls construct().

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
