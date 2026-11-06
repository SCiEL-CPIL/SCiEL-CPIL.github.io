import time
import random
from itertools import combinations_with_replacement
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    best_set = []
    best_len = 0
    
    def is_valid_addition(S, x, sums_set):
        # S is the current list, x is candidate
        # sums_set is the set of all triple sums of S
        # We need to check if adding x creates duplicate sums
        
        new_sums = set()
        
        # 1. x, x, x
        s = x + x + x
        if s in sums_set or s in new_sums:
            return False, None
        new_sums.add(s)
        
        # 2. x, x, y for y in S
        for y in S:
            s = x + x + y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
            
        # 3. x, y, z for y, z in S, y <= z
        k = len(S)
        for i in range(k):
            y = S[i]
            for j in range(i, k):
                z = S[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False, None
                new_sums.add(s)
                
        return True, new_sums

    def run_greedy(order):
        S = []
        sums_set = set()
        for x in order:
            if time.time() > deadline:
                break
            valid, new_sums = is_valid_addition(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
        return S

    def local_search(S, max_swaps=100):
        """Try to improve S by swapping elements"""
        if len(S) == 0:
            return S
        
        S = list(S)
        sums_set = set()
        # Rebuild sums_set
        for i in range(len(S)):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    sums_set.add(S[i] + S[j] + S[k])
        
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps:
            improved = False
            swap_count += 1
            if time.time() > deadline:
                break
            
            # Try swapping each element with a random candidate from 1..4096
            random.shuffle(S)
            for idx in range(len(S)):
                if time.time() > deadline:
                    break
                old_val = S[idx]
                # Try several random replacements
                for _ in range(20):
                    new_val = random.randint(1, 4096)
                    if new_val == old_val or new_val in S:
                        continue
                    
                    # Check if replacing old_val with new_val keeps validity
                    # Remove all sums involving old_val
                    temp_sums = set()
                    for i in range(len(S)):
                        for j in range(i, len(S)):
                            for k in range(j, len(S)):
                                s = S[i] + S[j] + S[k]
                                if old_val not in (S[i], S[j], S[k]):
                                    temp_sums.add(s)
                    
                    # Now check if new_val can be added
                    S[idx] = new_val
                    valid, _ = is_valid_addition(S[:idx] + S[idx+1:], new_val, temp_sums)
                    if valid:
                        # Accept the swap
                        S[idx] = new_val
                        # Rebuild full sums_set
                        sums_set = set()
                        for i in range(len(S)):
                            for j in range(i, len(S)):
                                for k in range(j, len(S)):
                                    sums_set.add(S[i] + S[j] + S[k])
                        improved = True
                        break
                    else:
                        S[idx] = old_val
                
                if improved:
                    break
        
        return S

    # 1. Standard Greedy (Increasing)
    order_inc = range(1, 4097)
    S_inc = run_greedy(order_inc)
    if len(S_inc) > best_len:
        best_set = S_inc
        best_len = len(S_inc)
    
    # Local search on increasing greedy
    if len(S_inc) > 0 and time.time() < deadline:
        S_inc_ls = local_search(S_inc, max_swaps=50)
        if len(S_inc_ls) > best_len:
            best_set = S_inc_ls
            best_len = len(S_inc_ls)
    
    # 2. Standard Greedy (Decreasing)
    order_dec = range(4096, 0, -1)
    S_dec = run_greedy(order_dec)
    if len(S_dec) > best_len:
        best_set = S_dec
        best_len = len(S_dec)
    
    # Local search on decreasing greedy
    if len(S_dec) > 0 and time.time() < deadline:
        S_dec_ls = local_search(S_dec, max_swaps=50)
        if len(S_dec_ls) > best_len:
            best_set = S_dec_ls
            best_len = len(S_dec_ls)
    
    # 3. Geometric spacing greedy
    # Use exponentially spaced candidates
    geo_candidates = []
    for i in range(1, 4097):
        # Add some geometrically spaced values
        geo_candidates.append(int(i ** 0.7))
    geo_candidates = sorted(set(geo_candidates))
    S_geo = run_greedy(geo_candidates)
    if len(S_geo) > best_len:
        best_set = S_geo
        best_len = len(S_geo)
    
    # 4. Randomized Greedy with diverse strategies
    rng = random.Random(42)
    
    while time.time() < deadline:
        strategy = rng.random()
        
        if strategy < 0.3:
            # Random permutation
            order = list(range(1, 4097))
            rng.shuffle(order)
        elif strategy < 0.5:
            # Biased toward smaller numbers (power law)
            order = []
            for i in range(4096):
                # Sample from power law distribution
                u = rng.random()
                val = int(1 + 4095 * u ** 2)  # Bias toward smaller
                order.append(val)
            order = sorted(set(order))
        elif strategy < 0.7:
            # Stratified: divide range into bins, sample from each
            order = []
            n_bins = 64
            bin_size = 4096 // n_bins
            for b in range(n_bins):
                lo = b * bin_size + 1
                hi = min((b + 1) * bin_size, 4096)
                # Sample 2-3 from each bin
                n_samples = min(3, hi - lo + 1)
                samples = rng.sample(range(lo, hi + 1), n_samples)
                order.extend(samples)
            order = sorted(set(order))
        elif strategy < 0.85:
            # Modular approach: pick residues
            modulus = rng.choice([7, 11, 13, 17, 19])
            residues = rng.sample(range(modulus), min(3, modulus))
            order = []
            for r in residues:
                order.extend(range(r if r > 0 else modulus, 4097, modulus))
            order = sorted(set(order))
        else:
            # Random with some structure: pick a random step size
            step = rng.randint(1, 50)
            start = rng.randint(1, step)
            order = list(range(start, 4097, step))
            # Add some random extras
            for _ in range(100):
                order.append(rng.randint(1, 4096))
            order = sorted(set(order))
        
        S_rand = run_greedy(order)
        if len(S_rand) > best_len:
            best_set = S_rand
            best_len = len(S_rand)
        
        # Local search on random results if they're decent
        if len(S_rand) >= best_len - 1 and len(S_rand) > 0 and time.time() < deadline:
            S_rand_ls = local_search(S_rand, max_swaps=20)
            if len(S_rand_ls) > best_len:
                best_set = S_rand_ls
                best_len = len(S_rand_ls)
    
    return best_set
