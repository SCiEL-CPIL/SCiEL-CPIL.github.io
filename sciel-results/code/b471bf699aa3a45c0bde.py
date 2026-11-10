import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    # Initial best: powers of 2, length 9
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_len = len(best_seq)
    
    def is_valid(seq):
        """Check if all consecutive block sums are distinct."""
        n = len(seq)
        if n <= 1:
            return True
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + seq[i]
        sums = set()
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                s = prefix[j] - prefix[i]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    if not is_valid(best_seq):
        best_seq = [1]
        best_len = 1
    
    def try_search(initial_diffs, deadline):
        """
        DFS search using difference-vector representation.
        markers: sorted list of prefix positions [0, d1, d1+d2, ...]
        A new difference d is valid if (last_marker + d - marker[i]) is not in diff_set for all i
        and last_marker + d <= 256
        """
        nonlocal best_seq, best_len
        
        if not initial_diffs:
            return
        
        # Validate initial sequence
        seq_from_diffs = list(initial_diffs)
        if not is_valid(seq_from_diffs):
            return
        
        if len(seq_from_diffs) > best_len:
            best_seq = seq_from_diffs[:]
            best_len = len(seq_from_diffs)
        
        # Build initial markers and diff set
        markers = [0]
        for d in initial_diffs:
            markers.append(markers[-1] + d)
        
        diff_set = set()
        n_init = len(markers)
        for i in range(n_init):
            for j in range(i + 1, n_init):
                diff_set.add(markers[j] - markers[i])
        
        # DFS
        def dfs():
            nonlocal best_seq, best_len
            
            if time.time() > deadline:
                return
            
            n = len(markers)
            last_marker = markers[-1]
            
            # Pruning: even with minimum diffs (1 each), can we beat best?
            # Max additional elements: at most 256 - last_marker (each diff >= 1)
            # But we also need the new marker <= 256
            max_additional = 256 - last_marker
            if n - 1 + max_additional <= best_len:
                return
            
            # Try adding each possible next difference
            # Minimum next diff is 1 (strictly increasing means diffs can be any positive int)
            # But we need last_marker + d <= 256, so d <= 256 - last_marker
            max_d = 256 - last_marker
            if max_d < 1:
                return
            
            for d in range(1, max_d + 1):
                if time.time() > deadline:
                    return
                
                new_marker = last_marker + d
                
                # Check: new_marker - markers[i] must not be in diff_set for any i
                ok = True
                for i in range(n):
                    diff = new_marker - markers[i]
                    if diff in diff_set:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                # Valid! Add it
                markers.append(new_marker)
                added_diffs = []
                for i in range(n):
                    diff = new_marker - markers[i]
                    diff_set.add(diff)
                    added_diffs.append(diff)
                
                # Update best
                if len(markers) - 1 > best_len:
                    seq = [markers[i + 1] - markers[i] for i in range(len(markers) - 1)]
                    best_seq = seq[:]
                    best_len = len(seq)
                
                dfs()
                
                # Undo
                markers.pop()
                for diff in added_diffs:
                    diff_set.discard(diff)
        
        dfs()
    
    # Phase 1: Deterministic search from powers of 2
    try_search([1, 2, 4, 8, 16, 32, 64, 128, 256], time_limit)
    
    # Phase 2: Randomized restarts with different initial sequences
    # Generate random strictly increasing sequences and try to extend them
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2.0:
        # Try several random starting points
        random.seed(42)
        num_restarts = min(20, int(remaining_time * 2))  # ~2 restarts per second
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 1.0:
                break
            
            # Generate a random initial sequence of length 4-6
            init_len = random.randint(4, 6)
            init_diffs = []
            current = 0
            for i in range(init_len):
                # Choose random diff, ensuring total stays <= 256
                max_remaining = 256 - current
                if max_remaining <= 1:
                    break
                # Biased toward smaller diffs for more room
                d = random.randint(1, min(max_remaining, 10 + i * 3))
                init_diffs.append(d)
                current += d
            
            if len(init_diffs) < 2:
                continue
            
            # Validate before searching
            seq_check = init_diffs[:]
            if is_valid(seq_check):
                if len(seq_check) > best_len:
                    best_seq = seq_check[:]
                    best_len = len(seq_check)
                try_search(init_diffs, time_limit)
    
    # Final validation
    if not is_valid(best_seq):
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return best_seq
