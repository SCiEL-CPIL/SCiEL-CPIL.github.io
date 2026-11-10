import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 512
    elements = list(range(1, N + 1))
    n = N
    
    # Precompute compatibility bitsets
    # compat[i] is a bitset where bit j is set iff lcm(elements[i], elements[j]) <= N
    # We use Python ints as arbitrary-precision bitsets
    compat = [0] * n
    for i in range(n):
        a = elements[i]
        bits = 0
        for j in range(n):
            b = elements[j]
            if a // gcd(a, b) * b <= N:
                bits |= (1 << j)
        compat[i] = bits
    
    # Precompute divisors
    def get_divisors(num):
        divs = set()
        i = 1
        while i * i <= num:
            if num % i == 0:
                divs.add(i)
                divs.add(num // i)
            i += 1
        return divs
    
    # Find best divisor set
    best_set = set()
    best_size = 0
    for L in range(1, N + 1):
        divs = get_divisors(L)
        if len(divs) > best_size:
            best_size = len(divs)
            best_set = divs
    
    # Convert set to bitset
    def set_to_bits(s):
        bits = 0
        for x in s:
            bits |= (1 << (x - 1))
        return bits
    
    def bits_to_set(bits):
        s = set()
        for i in range(n):
            if bits & (1 << i):
                s.add(elements[i])
        return s
    
    # Check if x can be added to set represented by bitset
    # x can be added iff all elements in the set are compatible with x
    # i.e., bits & ~compat[idx] == 0
    def can_add_bitset(x, bits):
        idx = x - 1
        return (bits & ~compat[idx]) == 0
    
    # Greedy expand using bitsets with scored ordering
    def greedy_expand(bits, max_iter=None):
        if max_iter is None:
            max_iter = 2000
        iter_count = 0
        changed = True
        while changed and iter_count < max_iter and time.time() < deadline:
            changed = False
            iter_count += 1
            # Find candidates not in current set
            remaining = ~bits & ((1 << n) - 1)
            if remaining == 0:
                break
            
            # Score candidates by number of compatible elements in current set
            # (more compatible = more likely to be addable, and we prefer smaller numbers for diversity)
            candidates = []
            temp = remaining
            while temp:
                # Get lowest set bit
                lb = temp & (-temp)
                idx = lb.bit_length() - 1
                x = elements[idx]
                # Check if addable
                if (bits & ~compat[idx]) == 0:
                    # Score: prefer elements with fewer conflicts with remaining (more flexibility)
                    # Actually, just add it - order doesn't matter much for bitset approach
                    candidates.append(x)
                temp ^= lb
            
            if not candidates:
                break
            
            # Pick a random candidate to add (randomization helps escape local optima)
            x = random.choice(candidates)
            bits |= (1 << (x - 1))
            changed = True
        
        return bits
    
    # Better greedy: process all addable candidates in one pass, repeat
    def greedy_expand_fast(bits):
        changed = True
        passes = 0
        while changed and passes < 100 and time.time() < deadline:
            changed = False
            passes += 1
            temp = ~bits & ((1 << n) - 1)
            while temp:
                lb = temp & (-temp)
                idx = lb.bit_length() - 1
                x = elements[idx]
                if (bits & ~compat[idx]) == 0:
                    bits |= (1 << idx)
                    changed = True
                temp ^= lb
        return bits
    
    # Local search: remove one element, re-expand
    def local_search(bits):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_size = bits.bit_count()
            # Try removing each element
            temp = bits
            while temp:
                lb = temp & (-temp)
                idx = lb.bit_length() - 1
                x = elements[idx]
                
                trial = bits & ~(1 << idx)
                expanded = greedy_expand_fast(trial)
                
                if expanded.bit_count() > current_size:
                    bits = expanded
                    improved = True
                    break
                temp ^= lb
        return bits
    
    # 2-swap local search: remove two, re-expand
    def local_search_2swap(bits):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_size = bits.bit_count()
            # Try pairs of elements to remove
            elements_list = bits_to_set(bits)
            el_list = list(elements_list)
            for i in range(len(el_list)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(el_list)):
                    if time.time() > deadline:
                        break
                    a, b = el_list[i], el_list[j]
                    trial = bits & ~(1 << (a - 1)) & ~(1 << (b - 1))
                    expanded = greedy_expand_fast(trial)
                    if expanded.bit_count() > current_size:
                        bits = expanded
                        improved = True
                        break
                if improved:
                    break
        return bits
    
    # Start from best divisor set
    current_bits = set_to_bits(best_set)
    current_bits = greedy_expand_fast(current_bits)
    
    # Local search 1-swap
    current_bits = local_search(current_bits)
    
    # Local search 2-swap
    current_bits = local_search_2swap(current_bits)
    
    # Random restarts with bitset approach
    best_bits = current_bits
    best_size = current_bits.bit_count()
    
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random start: pick random small numbers
        random_bits = 0
        for x in range(1, 80):
            if random.random() < 0.4:
                idx = x - 1
                if (random_bits & ~compat[idx]) == 0:
                    random_bits |= (1 << idx)
        
        # Greedy expand
        expanded = greedy_expand_fast(random_bits)
        # Quick local search
        expanded = local_search(expanded)
        
        if expanded.bit_count() > best_size:
            best_size = expanded.bit_count()
            best_bits = expanded
    
    # Final polish: try adding any remaining compatible elements
    final_bits = greedy_expand_fast(best_bits)
    final_bits = local_search(final_bits)
    
    return list(bits_to_set(final_bits))
