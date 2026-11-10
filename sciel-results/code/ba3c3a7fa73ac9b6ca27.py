import time

def construct():
    # The problem asks for 12 distinct positive integers with distinct subset sums.
    # The optimal solution is the powers of 2.
    # We can generate this using a greedy approach where each new element
    # is set to one more than the maximum possible sum of the previous elements.
    # This ensures all subset sums are distinct and contiguous from 0 to sum(elements).
    
    elements = []
    current_max_sum = 0
    
    for i in range(12):
        # The smallest integer that doesn't create a collision
        # is current_max_sum + 1.
        # Proof: If x > current_max_sum, then x + s > x > current_max_sum >= s for all s in S.
        # Thus the new sums {x+s} are all strictly greater than the old sums S.
        # Since S has distinct elements, x+S also has distinct elements.
        next_val = current_max_sum + 1
        elements.append(next_val)
        current_max_sum += next_val
        
    return elements
