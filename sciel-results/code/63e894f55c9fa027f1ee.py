"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = 256
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    prime_to_bit = {p: (1 << i) for i, p in enumerate(primes)}
    
    # Precompute prime factor bitmask for each number 1..256
    factor_mask = [0] * (n + 1)
    factor_mask[1] = 0  # 1 has no prime factors
    for num in range(2, n + 1):
        x = num
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= prime_to_bit[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= prime_to_bit[x]
        factor_mask[num] = mask
    
    # All candidate numbers 1..256
    candidates = list(range(1, n + 1))
    
    def is_coprime_set(selected):
        """Check if all pairs in selected are coprime using bitmask."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if factor_mask[selected[i]] & factor_mask[selected[j]]:
                    return False
        return True
    
    def get_sum(selected):
        return sum(selected)
    
    # Greedy initialization with different orderings
    def greedy_init(ordering):
        """Greedy: add numbers in given order if coprime to all selected."""
        selected = set()
        selected_masks = []
        for num in ordering:
            mask = factor_mask[num]
            ok = True
            for sm in selected_masks:
                if mask & sm:
                    ok = False
                    break
            if ok:
                selected.add(num)
                selected_masks.append(mask)
        return selected, selected_masks
    
    # Multiple orderings for diverse initial solutions
    orderings = [
        sorted(candidates, reverse=True),  # Largest first
        sorted(candidates, key=lambda x: -math.log(x)),  # By log (similar)
        sorted(candidates, key=lambda x: (factor_mask[x].bit_count(), -x)),  # Fewest factors first
        sorted(candidates, reverse=True),  # Try again largest first
        list(reversed(candidates)),  # Smallest first
        sorted(candidates, key=lambda x: (-x, factor_mask[x].bit_count())),  # Largest, then fewest factors
    ]
    
    best_solution = None
    best_sum = 0
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        selected, selected_masks = greedy_init(ordering)
        s = get_sum(selected)
        if s > best_sum:
            best_sum = s
            best_solution = set(selected)
    
    # Local search on the best solution
    if best_solution is None:
        # Fallback: just return [1]
        return [1]
    
    selected = set(best_solution)
    selected_masks = {num: factor_mask[num] for num in selected}
    current_sum = best_sum
    
    # Build lookup for quick mask access
    def can_add(num, sel_set, sel_masks):
        mask = factor_mask[num]
        for sm in sel_masks.values():
            if mask & sm:
                return False
        return True
    
    # Local search: 1-for-1 swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for rem in list(selected):
            if time.time() > deadline:
                break
            rem_mask = factor_mask[rem]
            new_sum = current_sum - rem
            # Try adding each non-selected element
            for add in range(1, n + 1):
                if add in selected:
                    continue
                add_mask = factor_mask[add]
                # Check if add conflicts with any remaining selected
                ok = True
                for sel_num, sel_mask in selected_masks.items():
                    if sel_num == rem:
                        continue
                    if add_mask & sel_mask:
                        ok = False
                        break
                if ok and new_sum + add > current_sum:
                    # Perform swap
                    selected.remove(rem)
                    del selected_masks[rem]
                    selected.add(add)
                    selected_masks[add] = add_mask
                    current_sum = new_sum + add
                    improved = True
                    break
    
    # 1-for-2 swaps: remove one, add two
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for rem in list(selected):
            if time.time() > deadline:
                break
            rem_mask = factor_mask[rem]
            new_sum = current_sum - rem
            # Try adding pairs
            for add1 in range(1, n + 1):
                if add1 in selected:
                    continue
                add1_mask = factor_mask[add1]
                # Check add1 against remaining
                ok1 = True
                for sel_num, sel_mask in selected_masks.items():
                    if sel_num == rem:
                        continue
                    if add1_mask & sel_mask:
                        ok1 = False
                        break
                if not ok1:
                    continue
                for add2 in range(add1 + 1, n + 1):
                    if add2 in selected:
                        continue
                    add2_mask = factor_mask[add2]
                    # Check add2 against remaining and add1
                    ok2 = True
                    if add1_mask & add2_mask:
                        ok2 = False
                    else:
                        for sel_num, sel_mask in selected_masks.items():
                            if sel_num == rem:
                                continue
                            if add2_mask & sel_mask:
                                ok2 = False
                                break
                    if ok2 and new_sum + add1 + add2 > current_sum:
                        # Perform 1-for-2 swap
                        selected.remove(rem)
                        del selected_masks[rem]
                        selected.add(add1)
                        selected_masks[add1] = add1_mask
                        selected.add(add2)
                        selected_masks[add2] = add2_mask
                        current_sum = new_sum + add1 + add2
                        improved = True
                        break
                if improved:
                    break
    
    # 2-for-1 swaps: remove two, add one
    improved = True
    while improved and time.time() < deadline:
        improved = False
        sel_list = list(selected)
        for i in range(len(sel_list)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(sel_list)):
                if time.time() > deadline:
                    break
                rem1 = sel_list[i]
                rem2 = sel_list[j]
                new_sum = current_sum - rem1 - rem2
                # Try adding each non-selected element
                for add in range(1, n + 1):
                    if add in selected:
                        continue
                    add_mask = factor_mask[add]
                    # Check if add conflicts with any remaining selected
                    ok = True
                    for sel_num, sel_mask in selected_masks.items():
                        if sel_num == rem1 or sel_num == rem2:
                            continue
                        if add_mask & sel_mask:
                            ok = False
                            break
                    if ok and new_sum + add > current_sum:
                        # Perform 2-for-1 swap
                        selected.remove(rem1)
                        del selected_masks[rem1]
                        selected.remove(rem2)
                        del selected_masks[rem2]
                        selected.add(add)
                        selected_masks[add] = add_mask
                        current_sum = new_sum + add
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # 2-for-2 swaps: remove two, add two
    improved = True
    while improved and time.time() < deadline:
        improved = False
        sel_list = list(selected)
        for i in range(len(sel_list)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(sel_list)):
                if time.time() > deadline:
                    break
                rem1 = sel_list[i]
                rem2 = sel_list[j]
                new_sum = current_sum - rem1 - rem2
                # Try adding pairs
                for add1 in range(1, n + 1):
                    if add1 in selected:
                        continue
                    add1_mask = factor_mask[add1]
                    # Check add1 against remaining
                    ok1 = True
                    for sel_num, sel_mask in selected_masks.items():
                        if sel_num == rem1 or sel_num == rem2:
                            continue
                        if add1_mask & sel_mask:
                            ok1 = False
                            break
                    if not ok1:
                        continue
                    for add2 in range(add1 + 1, n + 1):
                        if add2 in selected:
                            continue
                        add2_mask = factor_mask[add2]
                        # Check add2 against remaining, add1
                        ok2 = True
                        if add1_mask & add2_mask:
                            ok2 = False
                        else:
                            for sel_num, sel_mask in selected_masks.items():
                                if sel_num == rem1 or sel_num == rem2:
                                    continue
                                if add2_mask & sel_mask:
                                    ok2 = False
                                    break
                        if ok2 and new_sum + add1 + add2 > current_sum:
                            # Perform 2-for-2 swap
                            selected.remove(rem1)
                            del selected_masks[rem1]
                            selected.remove(rem2)
                            del selected_masks[rem2]
                            selected.add(add1)
                            selected_masks[add1] = add1_mask
                            selected.add(add2)
                            selected_masks[add2] = add2_mask
                            current_sum = new_sum + add1 + add2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Final verification and return
    result = sorted(selected)
    
    # Verify all pairs are coprime
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback to safe solution
                return [1]
    
    # Ensure all are Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
