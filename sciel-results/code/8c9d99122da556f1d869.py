def construct():
    n = 12
    seq = []
    current_sum = 0
    
    for i in range(n):
        # The next element must be greater than the sum of all previous elements
        # to ensure that any subset sum including this element is strictly greater
        # than any subset sum excluding it (which is bounded by current_sum).
        # To minimize the largest element, we pick the smallest valid integer.
        min_next_val = current_sum + 1
        seq.append(min_next_val)
        current_sum += min_next_val
        
    return seq
