import time

def construct():
    max_val = 256
    max_size = 60
    start_time = time.time()
    
    def try_add(sums_bitsets, new_elem, current_size):
        for k in range(1, current_size + 2):
            new_part = (sums_bitsets[k-1] << new_elem)
            for l in range(0, current_size + 1):
                if k == l:
                    continue
                if new_part & sums_bitsets[l]:
                    return (False, None)
        new_sums = list(sums_bitsets)
        for k in range(current_size, -1, -1):
            new_sums[k+1] |= (sums_bitsets[k] << new_elem)
        return (True, new_sums)
    
    def greedy(order):
        sums = [0] * (max_size + 1)
        sums[0] = 1
        result = []
        current_size = 0
        for elem in order:
            if current_size >= max_size:
                break
            if time.time() - start_time > 55:
                break
            success, new_sums = try_add(sums, elem, current_size)
            if success:
                sums = new_sums
                result.append(elem)
                current_size += 1
        return result
    
    # Try increasing order
    result = greedy(range(1, max_val + 1))
    
    return result
