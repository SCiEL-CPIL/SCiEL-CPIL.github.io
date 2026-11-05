import time
from math import gcd

def construct():
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin
    
    # Precompute prime factor masks
    # Primes up to 511
    primes = []
    is_prime = [True] * 512
    for i in range(2, 512):
        if is_prime[i]:
            primes.append(i)
            for j in range(2*i, 512, i):
                is_prime[j] = False
            
    # Map prime to bit index
    prime_to_bit = {p: 1 << i for i, p in enumerate(primes)}
    
    masks = [0] * 512
    for a in range(1, 512):
        m = 0
        temp = a
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                m |= prime_to_bit[p]
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            m |= prime_to_bit[temp]
        masks[a] = m
        
    # Greedy initialization
    chosen = []
    for a in range(511, 0, -1):
        if all((masks[a] & masks[b]) == 0 for b in chosen):
            chosen.append(a)
            
    current_score = sum(1.0 / (512 - a) for a in chosen)
    chosen_set = set(chosen)
    
    def is_coprime_to_all_except(a, except_set):
        ma = masks[a]
        for b in chosen:
            if b in except_set:
                continue
            if ma & masks[b]:
                return False
        return True
        
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # 1-for-1 swaps
        # Try replacing a_out with a_in
        # a_in should not be in chosen
        # a_out should be in chosen
        # Check if a_in is coprime to all in chosen \ {a_out}
        
        # Optimization: Precompute conflicts? 
        # Let's just do the checks.
        
        # To make it faster, we can iterate a_in in descending order (high value first)
        # and a_out in any order.
        
        found_improvement = False
        # 1-for-1
        for a_in in range(511, 0, -1):
            if a_in in chosen_set:
                continue
            for a_out in chosen:
                # Check coprimality
                ok = True
                ma = masks[a_in]
                for b in chosen:
                    if b == a_out:
                        continue
                    if ma & masks[b]:
                        ok = False
                        break
                if not ok:
                    continue
                
                new_score = current_score - 1.0/(512 - a_out) + 1.0/(512 - a_in)
                if new_score > current_score:
                    # Perform swap
                    chosen.remove(a_out)
                    chosen.append(a_in)
                    chosen_set.remove(a_out)
                    chosen_set.add(a_in)
                    current_score = new_score
                    improved = True
                    found_improvement = True
                    break # Restart 1-for-1 loop with new state? Or just continue? 
                    # Restarting is safer to ensure we don't miss things, but slower.
                    # Let's just break the inner loop and continue outer.
            if found_improvement:
                break
                
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # 2-for-1 swaps
        # Replace {a_out1, a_out2} with a_in
        found_improvement = False
        # To avoid O(N^3) which is 512^3 = 134M, we need to be careful.
        # But chosen size is small (~50). So pairs in chosen are ~1250.
        # a_in candidates ~460.
        # Total checks ~ 1250 * 460 = 575k.
        # Each check iterates over chosen (~50).
        # 575k * 50 = 28M ops. This is fine.
        
        for a_in in range(511, 0, -1):
            if a_in in chosen_set:
                continue
            ma = masks[a_in]
            # Iterate pairs in chosen
            for i in range(len(chosen)):
                a_out1 = chosen[i]
                for j in range(i+1, len(chosen)):
                    a_out2 = chosen[j]
                    
                    # Check if a_in is coprime to all in chosen \ {a_out1, a_out2}
                    ok = True
                    for b in chosen:
                        if b == a_out1 or b == a_out2:
                            continue
                        if ma & masks[b]:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    new_score = current_score - 1.0/(512 - a_out1) - 1.0/(512 - a_out2) + 1.0/(512 - a_in)
                    if new_score > current_score:
                        # Perform swap
                        chosen.remove(a_out1)
                        chosen.remove(a_out2)
                        chosen.append(a_in)
                        chosen_set.remove(a_out1)
                        chosen_set.remove(a_out2)
                        chosen_set.add(a_in)
                        current_score = new_score
                        improved = True
                        found_improvement = True
                        break
                if found_improvement:
                    break
            if found_improvement:
                break
                
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # 1-for-2 swaps?
        # Replace a_out with {a_in1, a_in2}
        # This increases cardinality.
        # Might be less common but worth trying if time permits.
        # Given the convexity, adding more large numbers is good.
        # But constraints are tight.
        
        # Let's add 2-for-2 as well if time permits.
        
        if time.time() - start_time > TIME_LIMIT - 20:
            # Stop adding complex moves, stick to 1-1 and 2-1
            continue

        # 2-for-2 swaps
        found_improvement = False
        for a_in1 in range(511, 0, -1):
            if a_in1 in chosen_set:
                continue
            ma1 = masks[a_in1]
            for a_in2 in range(a_in1 - 1, 0, -1):
                if a_in2 in chosen_set:
                    continue
                ma2 = masks[a_in2]
                if ma1 & ma2:
                    continue
                
                # Check if a_in1, a_in2 are coprime to all in chosen \ {a_out1, a_out2}
                # This requires iterating pairs of outs too.
                # 460 * 460 * 1250 * 50 = too slow.
                # Skip 2-for-2 for now or optimize.
                pass
                
    return chosen
