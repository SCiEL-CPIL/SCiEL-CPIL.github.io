import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Range of numbers
    N = 256
    numbers = list(range(2, N + 1))
    
    # Precompute conflicts for the "large" part [17, 256]
    # We will handle the "small" part [2, 16] separately as we can pick at most one.
    # Actually, let's just build the full conflict graph for [2, 256] if needed, 
    # but the structure suggests separating s.
    
    # Conflicts between y1, y2 in [17, 256]
    large_numbers = list(range(17, N + 1))
    large_conflicts = {y: set() for y in large_numbers}
    
    for i in range(len(large_numbers)):
        y1 = large_numbers[i]
        for j in range(i + 1, len(large_numbers)):
            y2 = large_numbers[j]
            if math.lcm(y1, y2) <= N:
                large_conflicts[y1].add(y2)
                large_conflicts[y2].add(y1)
                
    # Function to solve MWIS on a subset of large_numbers
    def solve_mwis(subset):
        if not subset:
            return set(), 0.0
            
        # Greedy initialization: add nodes with smallest index (largest weight) first
        # Weights are 1/y, so smaller y is better.
        sorted_subset = sorted(subset)
        
        current_set = set()
        score = 0.0
        
        for y in sorted_subset:
            # Check if y conflicts with any in current_set
            conflict = False
            for c in large_conflicts[y]:
                if c in current_set:
                    conflict = True
                    break
            if not conflict:
                current_set.add(y)
                score += 1.0 / y
                
        # Local Search
        improved = True
        while improved:
            improved = False
            # Try adding each node not in current_set
            candidates = [y for y in subset if y not in current_set]
            for y in candidates:
                if time.time() > deadline:
                    return current_set, score
                    
                # Find conflicts in current_set
                conflicts = [c for c in large_conflicts[y] if c in current_set]
                
                if not conflicts:
                    # Can add directly
                    current_set.add(y)
                    score += 1.0 / y
                    improved = True
                    break # Restart search
                    
                else:
                    # Check if adding y and removing conflicts is beneficial
                    loss = sum(1.0 / c for c in conflicts)
                    gain = 1.0 / y
                    if gain > loss:
                        for c in conflicts:
                            current_set.remove(c)
                        current_set.add(y)
                        score += gain - loss
                        improved = True
                        break # Restart search
                        
        return current_set, score

    best_score = -1.0
    best_set = set()
    
    # Try each small number s in [2, 16]
    for s in range(2, 17):
        if time.time() > deadline:
            break
            
        # Determine which large numbers are compatible with s
        compatible_large = []
        for y in large_numbers:
            if math.lcm(s, y) > N:
                compatible_large.append(y)
                
        # Solve MWIS for compatible_large
        subset_set = set(compatible_large)
        i_set, i_score = solve_mwis(subset_set)
        
        total_score = 1.0 / s + i_score
        total_set = {s} | i_set
        
        if total_score > best_score:
            best_score = total_score
            best_set = total_set
            
    # If no set found (should not happen), fallback to seed
    if not best_set:
        best_set = {2} | {y for y in range(129, 257) if y % 2 != 0}
        
    return list(best_set)

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}, Sum: {sum(1/x for x in result)}")
    # Quick validation
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.lcm(result[i], result[j]) <= 256:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
