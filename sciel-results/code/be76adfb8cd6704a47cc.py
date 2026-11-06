import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    # Fixed marks in 0-based: 0, 1, 4, 10 (corresponding to 1, 2, 5, 11 in 1-based)
    fixed_0based = frozenset({0, 1, 4, 10})
    max_val = 283  # 0-based, values in [0, 283]
    
    def get_diffs(marks):
        """Get all pairwise differences as a set"""
        diffs = set()
        n = len(marks)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(marks[j] - marks[i])
        return diffs
    
    def can_add(marks, diffs, x):
        """Check if x can be added to marks without creating duplicate diffs"""
        for m in marks:
            d = abs(x - m)
            if d in diffs:
                return False
        return True
    
    def add_mark(marks, diffs, x):
        """Add x to marks and update diffs, return new lists"""
        new_marks = sorted(marks + [x])
        new_diffs = diffs.copy()
        for m in marks:
            new_diffs.add(abs(x - m))
        return new_marks, new_diffs
    
    def greedy_build(fixed, rng):
        """Greedy construction with randomization"""
        marks = sorted(fixed)
        diffs = get_diffs(marks)
        
        # Generate candidates and shuffle
        candidates = [x for x in range(0, max_val + 1) if x not in fixed]
        rng.shuffle(candidates)
        
        for x in candidates:
            if can_add(marks, diffs, x):
                marks, diffs = add_mark(marks, diffs, x)
        
        return marks, diffs
    
    def local_search(marks, diffs, rng, max_iters=2000):
        """Local search with various moves"""
        current = sorted(marks)
        current_diffs = diffs.copy()
        current_size = len(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try inserting a new mark
                for _ in range(100):
                    x = rng.randint(0, max_val)
                    if x in current:
                        continue
                    if can_add(current, current_diffs, x):
                        current, current_diffs = add_mark(current, current_diffs, x)
                        current_size = len(current)
                        improved = True
                        break
            
            elif move_type < 0.7:
                # Try swapping: remove a non-fixed mark, add a different one
                non_fixed = [m for m in current if m not in fixed_0based]
                if non_fixed:
                    m_remove = rng.choice(non_fixed)
                    remaining = [m for m in current if m != m_remove]
                    remaining_diffs = get_diffs(remaining)
                    
                    for _ in range(50):
                        x = rng.randint(0, max_val)
                        if x in remaining:
                            continue
                        if can_add(remaining, remaining_diffs, x):
                            current = sorted(remaining + [x])
                            current_diffs = get_diffs(current)
                            current_size = len(current)
                            improved = True
                            break
            
            else:
                # Try moving a non-fixed mark to a new position
                non_fixed = [m for m in current if m not in fixed_0based]
                if non_fixed:
                    m_move = rng.choice(non_fixed)
                    remaining = [m for m in current if m != m_move]
                    remaining_diffs = get_diffs(remaining)
                    
                    for _ in range(50):
                        x = rng.randint(0, max_val)
                        if x in remaining:
                            continue
                        if can_add(remaining, remaining_diffs, x):
                            current = sorted(remaining + [x])
                            current_diffs = get_diffs(current)
                            current_size = len(current)
                            improved = True
                            break
        
        return current, current_diffs, current_size
    
    best = sorted(fixed_0based)
    best_diffs = get_diffs(best)
    best_size = len(best)
    
    rng = random.Random(12345)
    
    # Multiple restarts
    while time.time() < deadline:
        # Greedy build
        marks, diffs = greedy_build(fixed_0based, rng)
        
        if len(marks) > best_size:
            best = marks
            best_diffs = diffs
            best_size = len(marks)
        
        # Local search
        marks, diffs, size = local_search(marks, diffs, rng, max_iters=1000)
        
        if size > best_size:
            best = marks
            best_diffs = diffs
            best_size = size
        
        if best_size >= 20:
            break
    
    # Convert back to 1-based
    return [x + 1 for x in sorted(best)]
