import time
import random

def construct(n_rows, k):
    # 1. Greedy initialization (from seed)
    def greedy_construct():
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows

    initial_rows = greedy_construct()
    
    # 2. Cost Function
    def calculate_cost(rows):
        diffs = []
        max_val = 0
        for row in rows:
            if not row: continue
            max_val = max(max_val, row[-1])
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.append(row[j] - row[i])
        
        # Count conflicts
        freq = {}
        for d in diffs:
            freq[d] = freq.get(d, 0) + 1
        
        conflicts = sum(1 for count in freq.values() if count > 1)
        # Actually, we want the number of duplicate pairs? 
        # If a diff appears 3 times, it contributes to conflicts.
        # Standard metric: number of differences that are not unique.
        # Let's use: sum(max(0, count - 1)) for each diff value.
        # This counts how many "extra" occurrences there are.
        # If all unique, sum is 0.
        
        conflict_count = 0
        for count in freq.values():
            if count > 1:
                conflict_count += count - 1
                
        return conflict_count * 1000 + max_val

    # 3. Neighbor Generation
    def get_neighbor(rows):
        new_rows = [list(r) for r in rows]
        if random.random() < 0.2:
            # Re-randomize a row
            r_idx = random.randint(0, n_rows - 1)
            row = new_rows[r_idx]
            # Keep 0
            # Generate k-1 more numbers
            # To keep them somewhat close, we can sample increments
            # But we need strictly increasing.
            # Let's just pick random values in a range.
            # What range? 
            # Current max is roughly best_max.
            # Let's use a range up to 200.
            max_limit = 200
            row[0] = 0
            last = 0
            for j in range(1, k + 1):
                # Pick next value > last
                # Bias towards smaller values to keep max low?
                # Or uniform?
                # Uniform in [last+1, max_limit]
                if last + 1 >= max_limit:
                    # Stuck, just increment
                    row[j] = last + 1
                else:
                    row[j] = random.randint(last + 1, max_limit)
                last = row[j]
        else:
            # Change one element
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k) # 1 to k
            row = new_rows[r_idx]
            
            lower = row[j_idx - 1] + 1
            upper = row[j_idx + 1] - 1 if j_idx < k else 200
            
            if lower > upper:
                # Can't change this element, pick another
                return rows # Return original as neighbor (no change)
            
            # Sample new value
            # To encourage exploration, sample from a wider range
            # but respect bounds
            # Let's sample from [lower, upper]
            new_val = random.randint(lower, upper)
            row[j_idx] = new_val
            
        return new_rows

    # 4. Simulated Annealing
    best_rows = [list(r) for r in initial_rows]
    best_cost = calculate_cost(best_rows)
    
    current_rows = [list(r) for r in initial_rows]
    current_cost = best_cost
    
    start_time = time.time()
    time_limit = 110.0
    
    T = 100.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        neighbor = get_neighbor(current_rows)
        neighbor_cost = calculate_cost(neighbor)
        
        delta = neighbor_cost - current_cost
        
        if delta < 0:
            current_rows = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [list(r) for r in current_rows]
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < pow(2.718281828, -delta / T):
                current_rows = neighbor
                current_cost = neighbor_cost
                
        T *= cooling
        if T < 0.1:
            # Reheat? Or just stop cooling
            T = 1.0

    return best_rows
