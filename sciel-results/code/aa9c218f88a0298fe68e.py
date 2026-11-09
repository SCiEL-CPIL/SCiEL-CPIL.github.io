import numpy as np
import time
import random

# Precompute smallest prime factor for numbers up to 20000
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
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= factors_cache[s]
    return len(all_primes)

def find_clique_vectorized(target_primes, k=12, deadline=None, pool_size=1000):
    """
    Find k distinct numbers from 1..pool_size such that all pairwise sums are smooth w.r.t target_primes.
    """
    if deadline and time.time() > deadline:
        return None
    
    max_val = 2 * pool_size
    is_smooth = np.zeros(max_val + 1, dtype=bool)
    for i in range(2, max_val + 1):
        if factors_cache[i].issubset(target_primes):
            is_smooth[i] = True
            
    pool = np.arange(1, pool_size + 1)
    sums = pool[:, None] + pool[None, :]
    adj_matrix = is_smooth[sums]
    
    adj = [0] * pool_size
    for i in range(pool_size):
        row = adj_matrix[i]
        # Convert to string and then to int
        # row is boolean array
        # np.char.mod is not ideal for bool.
        # Let's use astype(int) then join
        # This creates a list of chars
        s = ''.join(map(str, row.astype(int)))
        # Reverse because int(s, 2) treats rightmost as LSB
        # row[0] is j=0. We want bit 0 to be set if row[0] is True.
        # So we need s to be "1" at the end if row[0] is True.
        # s is "010..." (row[0] is first char).
        # Reversed s is "...010".
        # If row[0] is True, s[0] is '1'. Reversed s[-1] is '1'.
        # int(..., 2) takes last char as LSB. So bit 0 is set. Correct.
        s_rev = s[::-1]
        adj[i] = int(s_rev, 2)
        
    found_clique = []
    
    def bron_kerbosch(R, P, X):
        if deadline and time.time() > deadline:
            return False
        if len(R) >= k:
            found_clique.extend(R[:k])
            return True
            
        if P == 0 and X == 0:
            return False
            
        if P | X == 0:
            return False
            
        all_nodes = P | X
        pivot = -1
        max_deg = -1
        temp = all_nodes
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            deg = (adj[idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = idx
            temp ^= lsb
            
        if pivot < 0:
            return False
            
        candidates_to_try = P & ~(adj[pivot])
        
        while candidates_to_try:
            lsb = candidates_to_try & (-candidates_to_try)
            v = lsb.bit_length() - 1
            candidates_to_try ^= lsb
            
            R.append(v)
            if bron_kerbosch(R, P & adj[v], X & adj[v]):
                R.pop()
                return True
            R.pop()
            P ^= lsb
            X |= lsb
            
        return False
        
    initial_P = (1 << pool_size) - 1
    if bron_kerbosch([], initial_P, 0):
        clique_indices = found_clique[:k]
        clique = [pool[i] for i in clique_indices]
        return clique
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 999
    best_arr = list(range(1, 13))
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for k in range(4, 9):
        if time.time() > deadline:
            break
            
        target_primes = frozenset(primes[:k])
        
        # Try pool sizes
        for pool_size in [1000, 2000]:
            if time.time() > deadline:
                break
                
            clique = find_clique_vectorized(target_primes, 12, deadline, pool_size)
            if clique:
                score = count_distinct_primes(clique)
                if score < best_score:
                    best_score = score
                    best_arr = clique[:]
                if score == k:
                    return best_arr

    # Fallback: Local search
    random.seed(42)
    current_arr = random.sample(range(1, 10001), 12)
    current_score = count_distinct_primes(current_arr)
    if current_score < best_score:
        best_score = current_score
        best_arr = current_arr[:]
        
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
