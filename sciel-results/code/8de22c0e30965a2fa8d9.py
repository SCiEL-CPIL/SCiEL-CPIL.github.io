def construct():
    n = 12
    # Use a boolean list to track which sums are already achieved
    # Max possible sum is 2^12 - 1 = 4095
    max_sum = 1 << n
    used_sums = [False] * max_sum
    used_sums[0] = True
    
    result = []
    current_max_sum = 0
    
    for _ in range(n):
        found = False
        # Try x from 1 upwards
        # The next x must be at least 1
        # Actually, we can start from 1, but we know it will be 2^k
        # To make it a "search" approach, we iterate x
        x = 1
        while not found:
            # Check if adding x creates any collision
            # New sums will be {s + x for s in used_sums if used_sums[s]}
            # Collision if any (s + x) is already in used_sums
            # And s + x must be < max_sum
            valid = True
            # Optimization: iterate through existing sums
            # Since used_sums is a list, we can iterate indices
            for s in range(current_max_sum + 1):
                if used_sums[s]:
                    new_s = s + x
                    if new_s >= max_sum or used_sums[new_s]:
                        valid = False
                        break
            
            if valid:
                # Accept x
                result.append(x)
                # Update used_sums
                # We need to add new sums
                # Iterate backwards or use a copy to avoid using newly added sums in the same step?
                # No, subset sums are formed by adding x to *existing* sums.
                # So we iterate over the *old* sums.
                # We can just loop s from 0 to current_max_sum
                for s in range(current_max_sum + 1):
                    if used_sums[s]:
                        used_sums[s + x] = True
                current_max_sum += x
                found = True
            else:
                x += 1
                
    return result
