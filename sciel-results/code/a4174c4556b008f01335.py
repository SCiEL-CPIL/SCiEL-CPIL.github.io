import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate candidate rows (Sidon sets of size k+1 starting at 0)
    # We generate up to a certain max value.
    # For k=5, size=6. 
    # Let's set M_gen based on k. 
    # A safe upper bound for M is roughly k^2 * n_rows? 
    # Actually, the counting bound for M is n_rows * k(k+1)/2.
    # Let's use M_gen = 2 * n_rows * k * (k+1) // 2 + 50
    M_gen = 2 * n_rows * k * (k + 1) // 2 + 50
    if M_gen < 100:
        M_gen = 100
        
    candidates = []
    
    def gen_sidon(current_set, used_diffs, max_elem_limit):
        if time.time() > deadline - 10:
            return
        if len(current_set) == k + 1:
            candidates.append((list(current_set), frozenset(used_diffs), max(current_set)))
            return
        
        last = current_set[-1] if current_set else -1
        for x in range(last + 1, max_elem_limit + 1):
            new_diffs = set()
            valid = True
            for y in current_set:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                gen_sidon(current_set + [x], used_diffs | new_diffs, max_elem_limit)
                if len(candidates) > 10000:
                    return

    gen_sidon([0], set(), M_gen)
    
    if not candidates:
        # Fallback: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    candidates.sort(key=lambda c: c[2])
    
    best_rows = None
    best_max_val = float('inf')
    
    while time.time() < deadline - 5:
        selected = []
        used_diffs = set()
        success = True
        
        for i in range(n_rows):
            compatible = []
            for c in candidates:
                if c[1] & used_diffs:
                    continue
                compatible.append(c)
            
            if not compatible:
                success = False
                break
            
            if random.random() < 0.3 and len(compatible) > 10:
                choice = random.choice(compatible[:10])
            else:
                choice = compatible[0]
            
            selected.append(choice)
            used_diffs |= choice[1]
        
        if success:
            current_max = max(c[2] for c in selected)
            if current_max < best_max_val:
                best_max_val = current_max
                best_rows = [c[0] for c in selected]
                
                # Local Search
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(n_rows):
                        others_diffs = set()
                        for j in range(n_rows):
                            if j != i:
                                others_diffs |= selected[j][1]
                        
                        others_max = max((selected[j][2] for j in range(n_rows) if j != i), default=0)
                        
                        for c in candidates:
                            if c is selected[i]:
                                continue
                            if c[2] >= best_max_val:
                                continue
                            if c[1] & others_diffs:
                                continue
                            
                            new_max = max(others_max, c[2])
                            if new_max < best_max_val:
                                selected[i] = c
                                best_max_val = new_max
                                improved = True
                                break
                    
                best_rows = [c[0] for c in selected]
                
    if best_rows:
        return best_rows
    else:
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
