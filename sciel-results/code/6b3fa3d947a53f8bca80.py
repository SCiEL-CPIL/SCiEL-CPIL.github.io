"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0  # safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80
    
    def valid_triple_sums(arr):
        """Check if all unordered triples (with repetition) have unique sums."""
        n = len(arr)
        if n == 0:
            return True
        # Generate all triple sums
        # For unordered triples with repetition: (i,j,k) with i<=j<=k
        sums = []
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.append(arr[i] + arr[j] + arr[k])
        return len(sums) == len(set(sums))
    
    def check_addition(selected, x):
        """Check if x can be added to selected without creating duplicate triple sums."""
        n = len(selected)
        existing_sums = set()
        # Compute all existing triple sums
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    existing_sums.add(selected[i] + selected[j] + selected[k])
        
        # New triple sums involving x:
        # (x, x, x), (x, x, y) for y in selected, (x, y, z) for y<=z in selected
        new_sums = [x + x + x]
        for y in selected:
            new_sums.append(x + x + y)
        for i in range(n):
            for j in range(i, n):
                new_sums.append(x + selected[i] + selected[j])
        
        # Check for duplicates among new sums
        if len(new_sums) != len(set(new_sums)):
            return False
        
        # Check for collisions with existing sums
        for s in new_sums:
            if s in existing_sums:
                return False
        
        return True
    
    def greedy_forward(start=1, max_size=MAX_SIZE):
        """Greedy: add smallest available number that maintains uniqueness."""
        selected = []
        triple_sums = set()
        
        for x in range(start, N + 1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 15:
                break
            
            n = len(selected)
            # Compute new triple sums involving x
            new_sums = [x + x + x]
            for y in selected:
                new_sums.append(x + x + y)
            for i in range(n):
                for j in range(i, n):
                    new_sums.append(x + selected[i] + selected[j])
            
            # Check for duplicates among new sums
            if len(new_sums) != len(set(new_sums)):
                continue
            
            # Check for collisions with existing sums
            valid = True
            for s in new_sums:
                if s in triple_sums:
                    valid = False
                    break
            
            if valid:
                selected.append(x)
                for s in new_sums:
                    triple_sums.add(s)
        
        return selected
    
    def greedy_backward(start=N, max_size=MAX_SIZE):
        """Greedy: add largest available number that maintains uniqueness."""
        selected = []
        triple_sums = set()
        
        for x in range(start, 0, -1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 15:
                break
            
            n = len(selected)
            new_sums = [x + x + x]
            for y in selected:
                new_sums.append(x + x + y)
            for i in range(n):
                for j in range(i, n):
                    new_sums.append(x + selected[i] + selected[j])
            
            if len(new_sums) != len(set(new_sums)):
                continue
            
            valid = True
            for s in new_sums:
                if s in triple_sums:
                    valid = False
                    break
            
            if valid:
                selected.append(x)
                for s in new_sums:
                    triple_sums.add(s)
        
        return selected
    
    def local_improve(selected):
        """Try to replace elements to fit more numbers."""
        if len(selected) >= MAX_SIZE:
            return selected
        if time.time() > DEADLINE - 10:
            return selected
        
        selected = sorted(selected)
        triple_sums = set()
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    triple_sums.add(selected[i] + selected[j] + selected[k])
        
        improved = True
        while improved and time.time() < DEADLINE - 10:
            improved = False
            n = len(selected)
            for idx in range(n):
                # Try replacing selected[idx] with a smaller or different number
                old_val = selected[idx]
                remaining = [v for i, v in enumerate(selected) if i != idx]
                
                # Recompute triple sums without old_val
                rem_sums = set()
                rn = len(remaining)
                for i in range(rn):
                    for j in range(i, rn):
                        for k in range(j, rn):
                            rem_sums.add(remaining[i] + remaining[j] + remaining[k])
                
                # Try candidates
                for x in range(1, N + 1):
                    if x == old_val or x in remaining:
                        continue
                    
                    new_sums = [x + x + x]
                    for y in remaining:
                        new_sums.append(x + x + y)
                    for i in range(rn):
                        for j in range(i, rn):
                            new_sums.append(x + remaining[i] + remaining[j])
                    
                    if len(new_sums) != len(set(new_sums)):
                        continue
                    
                    valid = True
                    for s in new_sums:
                        if s in rem_sums:
                            valid = False
                            break
                    
                    if valid:
                        # Replace
                        selected[idx] = x
                        selected.sort()
                        triple_sums = rem_sums.copy()
                        for s in new_sums:
                            triple_sums.add(s)
                        improved = True
                        break
        
        # Try adding more elements
        n = len(selected)
        for x in range(1, N + 1):
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 10:
                break
            if x in selected:
                continue
            
            new_sums = [x + x + x]
            for y in selected:
                new_sums.append(x + x + y)
            for i in range(n):
                for j in range(i, n):
                    new_sums.append(x + selected[i] + selected[j])
            
            if len(new_sums) != len(set(new_sums)):
                continue
            
            valid = True
            for s in new_sums:
                if s in triple_sums:
                    valid = False
                    break
            
            if valid:
                selected.append(x)
                selected.sort()
                for s in new_sums:
                    triple_sums.add(s)
        
        return selected
    
    # Strategy 1: Greedy forward
    best = greedy_forward()
    
    # Strategy 2: Greedy backward
    if time.time() < DEADLINE - 20:
        cand = greedy_backward()
        if len(cand) > len(best):
            best = cand
    
    # Strategy 3: Try different starting points for greedy forward
    if time.time() < DEADLINE - 20:
        for start in [2, 3, 5, 10]:
            if time.time() > DEADLINE - 15:
                break
            cand = greedy_forward(start=start)
            if len(cand) > len(best):
                best = cand
    
    # Local improvement
    if time.time() < DEADLINE - 10:
        best = local_improve(best)
    
    # Final validation
    if len(best) > 0:
        n = len(best)
        sums = []
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.append(best[i] + best[j] + best[k])
        if len(sums) != len(set(sums)):
            # Fallback to simple greedy
            best = greedy_forward()
    
    return best
# EVOLVE-BLOCK-END
