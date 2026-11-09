import time
import random
from itertools import combinations

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
factors_cache = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    n = i
    while n > 1:
        p = spf[n]
        factors_cache[i].add(p)
        while n % p == 0:
            n //= p

def count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= factors_cache[s]
    return len(all_primes)

def is_smooth(n, target_primes):
    """Check if n is smooth with respect to target_primes."""
    if n <= 1:
        return True
    return factors_cache[n].issubset(target_primes)

def generate_smooth_numbers(target_primes, limit=10000):
    """Generate all numbers <= limit that are smooth w.r.t target_primes."""
    smooth = []
    for num in range(1, limit + 1):
        if is_smooth(num, target_primes):
            smooth.append(num)
    return smooth

def find_clique_in_smooth_graph(target_primes, k=12, deadline=None, max_pool=5000, n_restarts=3):
    """
    Find k distinct numbers <= 10000 such that all pairwise sums are smooth w.r.t target_primes.
    Uses degree-ordered Bron-Kerbosch with bitsets.
    """
    if deadline and time.time() > deadline:
        return None
    
    # Generate all smooth candidates
    all_smooth = generate_smooth_numbers(target_primes)
    
    if len(all_smooth) < k:
        return None
    
    # If the pool is very large, we might need to sample, but for 7-smooth it's small.
    # For 11-smooth, it might be larger.
    # Let's cap the pool size to ensure graph construction is fast.
    # However, if we have a small pool, use all of it.
    
    best_clique = None
    
    # Strategy: Try the full set first if it's small enough (< 2000 nodes).
    # Otherwise, use random samples.
    
    pools_to_try = []
    
    if len(all_smooth) <= 2000:
        pools_to_try.append(all_smooth)
    else:
        # Sample a few large pools
        random.seed(42)
        for _ in range(3):
            sample_size = min(2000, len(all_smooth))
            pools_to_try.append(random.sample(all_smooth, sample_size))
            
        # Also try a "small numbers" pool
        pools_to_try.append(all_smooth[:2000])

    for pool in pools_to_try:
        if deadline and time.time() > deadline:
            break
            
        n_cand = len(pool)
        if n_cand < k:
            continue
            
        # Build adjacency bitsets
        adj = [0] * n_cand
        degrees = [0] * n_cand
        
        for i in range(n_cand):
            ci = pool[i]
            for j in range(i+1, n_cand):
                cj = pool[j]
                if is_smooth(ci + cj, target_primes):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    degrees[i] += 1
                    degrees[j] += 1
            
            if deadline and time.time() > deadline:
                return best_clique
        
        # Reorder nodes by degree (descending) to improve BK performance
        # Create list of (degree, index)
        order = sorted(range(n_cand), key=lambda x: degrees[x], reverse=True)
        
        # Map old index to new index
        old_to_new = [0] * n_cand
        for new_idx, old_idx in enumerate(order):
            old_to_new[old_idx] = new_idx
        
        # Rebuild adj bitsets in the new order
        new_adj = [0] * n_cand
        for old_i in range(n_cand):
            new_i = old_to_new[old_i]
            # Get bits from old_adj[old_i]
            bits = adj[old_i]
            # Map each set bit to new position
            # This is O(N^2) again, but N is small.
            # Optimization: Iterate through set bits
            temp_bits = bits
            while temp_bits:
                lsb = temp_bits & (-temp_bits)
                old_j = lsb.bit_length() - 1
                new_j = old_to_new[old_j]
                new_adj[new_i] |= (1 << new_j)
                temp_bits ^= lsb
                
        # Now run Bron-Kerbosch on new_adj
        found_clique = []
        
        def bron_kerbosch(R, P, X):
            if deadline and time.time() > deadline:
                return False
            
            # Pruning: if we can't reach k, return
            if len(R) + P.bit_count() < k:
                return False
                
            if len(R) >= k:
                found_clique.extend(R[:k])
                return True
            
            if P == 0 and X == 0:
                return False
                
            # Pivot selection
            all_nodes = P | X
            if all_nodes == 0:
                return False
                
            pivot = -1
            max_deg = -1
            temp = all_nodes
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                deg = (new_adj[idx] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = idx
                temp ^= lsb
            
            if pivot < 0:
                return False
                
            # Iterate over P \ N(pivot)
            candidates_to_try = P & ~(new_adj[pivot])
            
            while candidates_to_try:
                lsb = candidates_to_try & (-candidates_to_try)
                v = lsb.bit_length() - 1
                candidates_to_try ^= lsb
                
                R.append(v)
                if bron_kerbosch(R, P & new_adj[v], X & new_adj[v]):
                    R.pop()
                    return True
                R.pop()
                P ^= lsb
                X |= lsb
            
            return False
        
        initial_P = (1 << n_cand) - 1
        if bron_kerbosch([], initial_P, 0):
            # Map back to original values
            # found_clique contains new indices
            clique_vals = []
            for new_idx in found_clique[:k]:
                old_idx = order[new_idx]
                clique_vals.append(pool[old_idx])
            
            if len(set(clique_vals)) == k:
                return clique_vals
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline, 5s safety margin
    
    best_score = 999
    best_arr = list(range(1, 13))  # fallback
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Strategy 1: Try to find a clique in the smooth graph for small sets of primes.
    # Prioritize {2,3,5,7} as per knowledge.
    
    # Try k=4 first: {2,3,5,7}
    # Then k=5: {2,3,5,7,11}
    # etc.
    
    for k in range(4, 8):
        if time.time() > deadline:
            break
            
        # Option 1: First k primes
        target_primes = frozenset(primes[:k])
        
        clique = find_clique_in_smooth_graph(target_primes, 12, deadline, max_pool=3000, n_restarts=3)
        if clique:
            score = count_distinct_primes(clique)
            if score < best_score:
                best_score = score
                best_arr = clique[:]
            if score == k:
                return best_arr
                
        # Option 2: Random subsets of size k (if first k primes failed)
        random.seed(42 + k)
        for _ in range(3):
            if time.time() > deadline:
                break
            selected = random.sample(primes[:15], k)
            target_primes = frozenset(selected)
            clique = find_clique_in_smooth_graph(target_primes, 12, deadline, max_pool=2000, n_restarts=2)
            if clique:
                score = count_distinct_primes(clique)
                if score < best_score:
                    best_score = score
                    best_arr = clique[:]
                if score == k:
                    return best_arr

    # Strategy 2: Fallback to heuristic search if no perfect smooth clique found
    # Use powers of 2
    powers_of_2 = [1 << i for i in range(12)]
    if max(powers_of_2) <= 10000:
        score = count_distinct_primes(powers_of_2)
        if score < best_score:
            best_score = score
            best_arr = powers_of_2[:]
            
    # Multiples
    for mult in range(1, 20):
        if time.time() > deadline:
            return best_arr
        for start_k in range(1, 30):
            if time.time() > deadline:
                return best_arr
            arr = [mult * (start_k + i) for i in range(12)]
            if len(set(arr)) < 12 or max(arr) > 10000:
                continue
            score = count_distinct_primes(arr)
            if score < best_score:
                best_score = score
                best_arr = arr[:]
                
    # Local search refinement on best_arr
    if best_score > 4:
        random.seed(42)
        current_arr = best_arr[:]
        current_score = best_score
        
        temp = 10.0
        cooling = 0.99
        min_temp = 0.1
        
        while temp > min_temp and time.time() < deadline:
            idx = random.randint(0, 11)
            old_val = current_arr[idx]
            new_val = random.randint(1, 10000)
            
            if new_val in current_arr and new_val != old_val:
                continue
                
            current_arr[idx] = new_val
            score = count_distinct_primes(current_arr)
            
            delta = score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_score = score
                if score < best_score:
                    best_score = score
                    best_arr = current_arr[:]
            else:
                current_arr[idx] = old_val
                
            temp *= cooling

    return best_arr

if __name__ == "__main__":
    res = construct()
    print(res)
    print(count_distinct_primes(res))
