import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # Precompute: for each value v in 1..512, store a bitmask of which other values
    # form a 3-AP with v as the middle element (v-d, v, v+d)
    # We'll use a different approach: precompute all possible (a, d) pairs and check membership
    
    def evaluate(S):
        S_set = set(S)
        valid_diffs = set()
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = a + 2 * d
                if c <= 512 and c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def evaluate_fast(S_set):
        """Faster evaluation using direct membership checks."""
        valid_diffs = set()
        S_list = sorted(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                if d > 256:  # a+2d > 512
                    break
                c = a + 2 * d
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    # Precompute for each element x in 1..512, the list of (y, d) such that
    # x is the first element of a 3-AP: (x, x+d, x+2d)
    # And for each element x, the list of (y, d) such that x is the second element:
    # (x-d, x, x+d)
    # This allows targeted moves.
    
    def get_targeted_moves(S_set):
        """Generate moves that specifically try to complete 3-APs."""
        moves = []
        S_list = sorted(S_set)
        n = len(S_list)
        
        # For each pair (a, b) in S, check if a+2*(b-a) is missing -> add it
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = a + 2 * d
                if c <= 512 and c not in S_set:
                    # Adding c would complete a 3-AP with difference d
                    moves.append((a, c, d))  # remove a, add c? No, we want to add c and keep a,b
                    # Actually we need to remove something else and add c
                    break
        
        # For each pair (a, c) in S with c-a even, check if middle (a+c)//2 is missing
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                c = S_list[j]
                diff = c - a
                if diff % 2 == 0:
                    mid = (a + c) // 2
                    if mid not in S_set:
                        moves.append((a, mid, diff // 2))
                        break
        
        # For each element a in S and each d, check if a+d and a+2d are both missing
        # but that's too expensive. Instead, for each a in S, find d such that
        # exactly one of a+d, a+2d is in S, and the other is not.
        for a in S_list:
            for d in range(1, 257):
                b = a + d
                c = a + 2 * d
                if c > 512:
                    break
                b_in = b in S_set
                c_in = c in S_set
                if b_in and not c_in:
                    moves.append((a, c, d))
                elif not b_in and c_in:
                    moves.append((a, b, d))
        
        return moves
    
    # Known good initial set from parent (score 4.8)
    initial_set = set(list(range(1, 27)) + [28, 29, 30, 31, 32, 48])
    
    def random_set():
        return set(random.sample(range(1, 513), 32))
    
    def structured_set():
        base = random.randint(1, 400)
        s = set()
        block_size = random.randint(8, 16)
        for i in range(block_size):
            val = base + i * random.randint(1, 3)
            if 1 <= val <= 512 and val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        return s
    
    def arithmetic_progression_set():
        """Create a set based on an arithmetic progression with some perturbations."""
        base = random.randint(1, 300)
        step = random.randint(1, 5)
        s = set()
        for i in range(32):
            val = base + i * step
            if val > 512:
                break
            if val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        return s
    
    best_set = None
    best_score = 0
    
    candidates = [
        initial_set,
        set(range(1, 33)),
        random_set(),
        random_set(),
        structured_set(),
        structured_set(),
        arithmetic_progression_set(),
        arithmetic_progression_set(),
    ]
    
    for c in candidates:
        s = evaluate_fast(c)
        if s > best_score:
            best_score = s
            best_set = c.copy()
    
    current_set = best_set.copy()
    current_score = best_score
    
    no_improvement_count = 0
    all_elements = list(range(1, 513))
    
    while time.time() - start_time < timeout:
        improved = False
        steps_in_iteration = 0
        max_steps_per_iter = 5000  # Increased due to faster evaluation
        
        while steps_in_iteration < max_steps_per_iter:
            if time.time() - start_time > timeout - 5.0:
                break
            
            move_type = random.random()
            new_set = set(current_set)
            
            if move_type < 0.35:
                # Targeted move: try to complete a 3-AP
                moves = get_targeted_moves(current_set)
                if moves:
                    # Pick a random targeted move
                    a_val, add_val, d_val = random.choice(moves)
                    # We need to remove some element and add add_val
                    # Remove a random element that's not critical
                    removable = [x for x in current_set if x != add_val]
                    if removable:
                        x_to_remove = random.choice(removable)
                        new_set.remove(x_to_remove)
                        new_set.add(add_val)
                    else:
                        steps_in_iteration += 1
                        continue
                else:
                    # No targeted moves available, fall back to random
                    x_to_remove = random.choice(list(current_set))
                    y_to_add = random.choice(all_elements)
                    if y_to_add in current_set:
                        steps_in_iteration += 1
                        continue
                    new_set.remove(x_to_remove)
                    new_set.add(y_to_add)
            elif move_type < 0.7:
                # Random single swap
                x_to_remove = random.choice(list(current_set))
                y_to_add = random.choice(all_elements)
                if y_to_add in current_set:
                    steps_in_iteration += 1
                    continue
                new_set.remove(x_to_remove)
                new_set.add(y_to_add)
            else:
                # Double swap
                x1, x2 = random.sample(list(current_set), 2)
                y1 = random.choice(all_elements)
                y2 = random.choice(all_elements)
                if y1 in current_set or y2 in current_set or y1 == y2:
                    steps_in_iteration += 1
                    continue
                new_set.remove(x1)
                new_set.remove(x2)
                new_set.add(y1)
                new_set.add(y2)
            
            new_score = evaluate_fast(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                no_improvement_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set.copy()
            else:
                no_improvement_count += 1
            
            steps_in_iteration += 1
        
        if time.time() - start_time > timeout - 5.0:
            break
        
        if no_improvement_count > 300:
            # Restart from best with some randomization
            num_swaps = random.randint(3, 8)
            current_set = best_set.copy()
            for _ in range(num_swaps):
                x_to_remove = random.choice(list(current_set))
                y_to_add = random.choice(all_elements)
                if y_to_add not in current_set:
                    current_set.remove(x_to_remove)
                    current_set.add(y_to_add)
            current_score = evaluate_fast(current_set)
            no_improvement_count = 0
        
        if no_improvement_count > 1500:
            current_set = best_set.copy()
            current_score = best_score
            no_improvement_count = 0

    return list(best_set)
