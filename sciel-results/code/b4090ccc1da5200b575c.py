import time


def construct():
        start_time = time.time()
        deadline = start_time + 50  # 50s safety margin
        
        selected = []
        # Bit 0 is set initially (sum 0 is reachable)
        # We care about sums up to 200.
        # Mask to keep only bits 0 to 200
        MASK = (1 << 201) - 1
        reachable = 1
        
        for x in range(100, 0, -1):
            if time.time() > deadline:
                break
                
            # Shift reachable sums by x
            # (reachable << x) gives sums that include x
            # reachable | (reachable << x) gives all sums with or without x
            new_reachable = reachable | (reachable << x)
            
            # Check if sum 200 is reachable
            # Bit 200 corresponds to (1 << 200)
            if not (new_reachable & (1 << 200)):
                selected.append(x)
                # Update reachable, masking out sums > 200
                # Sums > 200 cannot contribute to sum 200 later (since we only add positive numbers)
                reachable = new_reachable & MASK
                
        return selected
