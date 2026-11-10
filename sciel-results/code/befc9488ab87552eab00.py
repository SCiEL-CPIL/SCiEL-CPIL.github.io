"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(256)
    prime_to_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    mask_to_max_val = {}
    
    for x in range(2, 257):
        mask = 0
        temp = x
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            if temp in prime_to_index:
                mask |= (1 << prime_to_index[temp])
        
        if mask == 0:
            continue
            
        if mask in mask_to_max_val:
            if x > mask_to_max_val[mask]:
                mask_to_max_val[mask] = x
        else:
            mask_to_max_val[mask] = x
            
    items = [(m, v) for m, v in mask_to_max_val.items()]
    
    prime_to_items = [[] for _ in range(num_primes)]
    for idx, (m, v) in enumerate(items):
        for i in range(num_primes):
            if m & (1 << i):
                prime_to_items[i].append(idx)
                
    memo = {}
    decisions = {}
    
    def solve(available_mask):
        if available_mask == 0:
            return 0
        if available_mask in memo:
            return memo[available_mask]
            
        best_prime_idx = -1
        min_options = float('inf')
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            options = prime_to_items[idx]
            valid_options = 0
            for item_idx in options:
                if items[item_idx][0] & ~available_mask == 0:
                    valid_options += 1
            if valid_options < min_options:
                min_options = valid_options
                best_prime_idx = idx
            temp_mask ^= lsb
            
        if best_prime_idx == -1:
            memo[available_mask] = 0
            decisions[available_mask] = -1
            return 0
            
        mask_without_p = available_mask & ~(1 << best_prime_idx)
        best_val = solve(mask_without_p)
        best_decision = -1
        
        for item_idx in prime_to_items[best_prime_idx]:
            item_mask, item_val = items[item_idx]
            if item_mask & ~available_mask == 0:
                new_mask = available_mask & ~item_mask
                val = item_val + solve(new_mask)
                if val > best_val:
                    best_val = val
                    best_decision = item_idx
                    
        memo[available_mask] = best_val
        decisions[available_mask] = best_decision
        return best_val

    all_primes_mask = (1 << num_primes) - 1
    max_sum = solve(all_primes_mask)
    
    selected_numbers = []
    current_mask = all_primes_mask
    while current_mask:
        item_idx = decisions[current_mask]
        if item_idx == -1:
            best_prime_idx = -1
            min_options = float('inf')
            temp_mask = current_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                options = prime_to_items[idx]
                valid_options = 0
                for i_idx in options:
                    if items[i_idx][0] & ~current_mask == 0:
                        valid_options += 1
                if valid_options < min_options:
                    min_options = valid_options
                    best_prime_idx = idx
                temp_mask ^= lsb
            current_mask = current_mask & ~(1 << best_prime_idx)
        else:
            item_mask, item_val = items[item_idx]
            selected_numbers.append(item_val)
            current_mask = current_mask & ~item_mask
            
    selected_numbers.append(1)
    
    return selected_numbers
# EVOLVE-BLOCK-END
