import time
import random

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 55
    MAX_SUM = 65537  # max possible block sum + 1

    def get_block_sums(seq):
        """Return set of all non-empty consecutive block sums."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        """Check if seq is strictly increasing and all consecutive block sums are distinct."""
        n = len(seq)
        if n == 0:
            return True
        for i in range(1, n):
            if seq[i] <= seq[i-1]:
                return False
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_init():
        """Greedy: pick smallest valid next value."""
        values = []
        used_sums = set()
        for x in range(1, N + 1):
            # check x alone
            if x in used_sums:
                continue
            # check all suffix sums with x prepended
            new_sums = {x}
            s = x
            valid = True
            for a in reversed(values):
                s += a
                new_sums.add(s)
                if s in used_sums:
                    valid = False
                    break
            if valid:
                values.append(x)
                used_sums.update(new_sums)
        return values

    # Initialize with greedy
    best_seq = greedy_init()
    best_len = len(best_seq)
    
    # Local search: try to insert elements or replace to improve
    def try_insert(seq, pos, val):
        """Try inserting val at position pos. Return new seq or None."""
        if pos < 0 or pos > len(seq):
            return None
        # Check strictly increasing
        if pos > 0 and seq[pos-1] >= val:
            return None
        if pos < len(seq) and val >= seq[pos]:
            return None
        new_seq = seq[:pos] + [val] + seq[pos:]
        return new_seq if is_valid(new_seq) else None

    def try_replace(seq, idx, val):
        """Try replacing seq[idx] with val. Return new seq or None."""
        if val <= 0 or val > N:
            return None
        # Check strictly increasing
        if idx > 0 and seq[idx-1] >= val:
            return None
        if idx < len(seq) - 1 and val >= seq[idx+1]:
            return None
        new_seq = seq[:idx] + [val] + seq[idx+1:]
        return new_seq if is_valid(new_seq) else None

    def local_search(seq, max_iters):
        """Run local search: try insertions and replacements."""
        current = list(seq)
        current_len = len(current)
        best_local = list(current)
        best_local_len = current_len
        used = set(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            action = random.random()
            improved = False
            
            if action < 0.5 and len(current) < N:
                # Try to insert a new element
                # Pick a random value not in current
                available = [v for v in range(1, N+1) if v not in used]
                if not available:
                    break
                val = random.choice(available)
                # Try random position
                pos = random.randint(0, len(current))
                new_seq = try_insert(current, pos, val)
                if new_seq is not None:
                    current = new_seq
                    used.add(val)
                    current_len += 1
                    if current_len > best_local_len:
                        best_local_len = current_len
                        best_local = list(current)
                    improved = True
            elif action < 0.8:
                # Try to replace an element with a smaller unused value
                if not current:
                    continue
                idx = random.randint(0, len(current)-1)
                old_val = current[idx]
                # Find unused values smaller than constraints allow
                lower = current[idx-1] + 1 if idx > 0 else 1
                upper = current[idx+1] - 1 if idx < len(current)-1 else N
                candidates = [v for v in range(lower, upper+1) if v not in used and v != old_val]
                if candidates:
                    val = random.choice(candidates)
                    new_seq = try_replace(current, idx, val)
                    if new_seq is not None:
                        used.remove(old_val)
                        used.add(val)
                        current = new_seq
                        improved = True
            else:
                # Try to remove an element (to free up space for better arrangement)
                if len(current) > 1:
                    idx = random.randint(0, len(current)-1)
                    removed = current[idx]
                    new_seq = current[:idx] + current[idx+1:]
                    if is_valid(new_seq):
                        current = new_seq
                        used.remove(removed)
                        current_len -= 1
                        improved = True
            
            # If no improvement, keep going (annealing-like: we don't accept worse here,
            # but we do allow removals to explore)
        
        return best_local, best_local_len

    # Run multiple rounds of local search with random restarts
    global_best = list(best_seq)
    global_best_len = best_len
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            start_seq = list(global_best)
        else:
            # Random restart: take global best and randomly perturb
            start_seq = list(global_best)
            # Randomly remove a few elements
            n_remove = random.randint(0, min(3, len(start_seq)//3))
            for _ in range(n_remove):
                if start_seq:
                    idx = random.randint(0, len(start_seq)-1)
                    start_seq.pop(idx)
        
        # Run local search from this starting point
        local_result, local_len = local_search(start_seq, max_iters=5000)
        
        if local_len > global_best_len:
            global_best_len = local_len
            global_best = local_result
    
    return global_best
