import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    best_seq = []
    best_len = 0
    
    # Greedy baseline
    def get_greedy():
        values, products = [], set()
        for x in range(2, 129):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values
    
    initial_seq = get_greedy()
    best_seq = initial_seq
    best_len = len(best_seq)
    
    # Precompute all numbers
    numbers = list(range(2, 129))
    
    def dfs(start_idx, seq, prods, suffixes):
        nonlocal best_seq, best_len, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if len(seq) > best_len:
            best_seq = seq.copy()
            best_len = len(seq)
            
        # Pruning
        # Max possible additional elements: 128 - start_idx (since start_idx is index in numbers)
        # Wait, start_idx is the index in the `numbers` list.
        # The next number must be >= numbers[start_idx].
        # So max additional is len(numbers) - start_idx.
        if len(seq) + (len(numbers) - start_idx) <= best_len:
            return
            
        for i in range(start_idx, len(numbers)):
            if time.time() - start_time > time_limit:
                return
                
            x = numbers[i]
            
            # Check if x is valid
            if x in prods:
                continue
                
            new_prods = [x]
            valid = True
            # suffixes contains products of suffixes of seq
            # seq = [s0, ..., sk-1]
            # suffixes = [sk-1, sk-2*sk-1, ...]
            for s in suffixes:
                p = x * s
                if p in prods:
                    valid = False
                    break
                new_prods.append(p)
                
            if not valid:
                continue
                
            # Recurse
            # New suffixes for seq + [x]
            # New suffixes: [x, x*sk-1, x*sk-2*sk-1, ...]
            # Which is exactly new_prods in reverse order?
            # new_prods = [x, x*s_{k-1}, x*s_{k-2}s_{k-1}, ...]
            # The suffixes of the new sequence are:
            # 1. x
            # 2. x * (last element of old seq) = x * s_{k-1}
            # 3. x * (last two elements of old seq) = x * s_{k-2} s_{k-1}
            # ...
            # So new_suffixes = new_prods reversed?
            # new_prods[0] = x
            # new_prods[1] = x * s_{k-1}
            # ...
            # The suffixes of the new sequence are indeed the elements of new_prods.
            # But they need to be in the order: [x, x*s_{k-1}, ...]
            # Wait, the definition of suffixes in the next call:
            # For a sequence S, suffixes[0] = S[-1], suffixes[1] = S[-2]*S[-1], etc.
            # If new_seq = seq + [x], then:
            # new_seq[-1] = x
            # new_seq[-2] = seq[-1]
            # So new_suffixes[0] = x
            # new_suffixes[1] = x * seq[-1]
            # new_suffixes[2] = x * seq[-1] * seq[-2]
            # This matches new_prods exactly.
            # So we can pass new_prods as the new suffixes.
            
            new_prods_set = prods.union(new_prods)
            
            dfs(i + 1, seq + [x], new_prods_set, new_prods)
            
    dfs(0, [], set(), [])
    
    return best_seq
