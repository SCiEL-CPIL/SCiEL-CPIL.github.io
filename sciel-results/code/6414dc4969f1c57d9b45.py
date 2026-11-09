"""
Combined Snake Construction for Q_9:
- Bidirectional Warnsdorff Growth (from Inspiration)
- Local Insertion Search (from Current Program)
- Bitmask Optimizations
"""
import time
import random

N = 9
MAX_VERTS = 2 ** N

# EVOLVE-BLOCK-START

def construct():
    """
    Construct the longest induced path (snake) in Q_9 within the time budget.
    Strategy:
    1. Bidirectional growth using Warnsdorff's rule (prioritize constrained ends).
    2. Local search via vertex insertion to extend stuck paths.
    3. Multiple randomized restarts.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_path = None
    best_len = 0
    
    def is_valid_tail(v, path, used_set):
        """Check if v can be appended to path."""
        if v in used_set:
            return False
        # v must not be adjacent to any vertex in path except the last one
        # path[-1] is the neighbor we are moving from, so we check path[:-1]
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
        return True

    def is_valid_head(v, path, used_set):
        """Check if v can be prepended to path."""
        if v in used_set:
            return False
        # v must not be adjacent to any vertex in path except the first one
        # path[0] is the neighbor we are moving from, so we check path[1:]
        for p in path[1:]:
            if (v ^ p).bit_count() == 1:
                return False
        return True

    def count_options_tail(cur, path, used_set):
        """Count valid moves from cur if it were the tail."""
        count = 0
        for b in range(N):
            v = cur ^ (1 << b)
            if is_valid_tail(v, path, used_set):
                count += 1
        return count

    def count_options_head(cur, path, used_set):
        """Count valid moves from cur if it were the head."""
        count = 0
        for b in range(N):
            v = cur ^ (1 << b)
            if is_valid_head(v, path, used_set):
                count += 1
        return count

    def warnsdorff_grow(path, used_set, rng, max_steps=500):
        """
        Bidirectional growth with Warnsdorff's rule.
        1. Determine valid moves for head and tail.
        2. Select the end with fewer total valid moves (more constrained).
        3. Among valid moves at that end, select the one with fewest subsequent options.
        """
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            tail_moves = []
            for b in range(N):
                v = path[-1] ^ (1 << b)
                if is_valid_tail(v, path, used_set):
                    # Lookahead: count options from v (assuming v becomes new tail)
                    # We need to simulate adding v to check options from v's perspective
                    # But v is adjacent to path[-1].
                    # For Warnsdorff, we count how many neighbors of v are valid *after* adding v.
                    # A neighbor w of v is valid if w is not in used_set U {v} and 
                    # w is not adjacent to any p in path[:-1] (since path[-1] is already adjacent to v, 
                    # but wait, w is adjacent to v, so w is at distance 2 from path[-1]? No.
                    # If we add v to tail:
                    # New tail is v.
                    # Next move from v must be w.
                    # w must not be in used_set U {v}.
                    # w must not be adjacent to any p in path[:-1].
                    # Note: w is adjacent to v. v is adjacent to path[-1].
                    # So w is at distance 2 from path[-1] (unless w == path[-1] which is in used).
                    # So we just check against path[:-1].
                    sub_count = 0
                    for b2 in range(N):
                        w = v ^ (1 << b2)
                        if w == path[-1]: # w is the vertex we came from, already in used
                            continue
                        if w in used_set:
                            continue
                        # Check if w is adjacent to any p in path[:-1]
                        valid_w = True
                        for p in path[:-1]:
                            if (w ^ p).bit_count() == 1:
                                valid_w = False
                                break
                        if valid_w:
                            sub_count += 1
                    tail_moves.append((sub_count, b, v))

            head_moves = []
            for b in range(N):
                v = path[0] ^ (1 << b)
                if is_valid_head(v, path, used_set):
                    sub_count = 0
                    for b2 in range(N):
                        w = v ^ (1 << b2)
                        if w == path[0]:
                            continue
                        if w in used_set:
                            continue
                        valid_w = True
                        for p in path[1:]:
                            if (w ^ p).bit_count() == 1:
                                valid_w = False
                                break
                        if valid_w:
                            sub_count += 1
                    head_moves.append((sub_count, b, v))

            if not tail_moves and not head_moves:
                break

            # Decide which end to grow
            # If one end is stuck, grow the other.
            # Otherwise, grow the end with fewer total moves (min len).
            # Tie-break with randomness.
            if not tail_moves:
                choose_head = True
            elif not head_moves:
                choose_head = False
            else:
                # Warnsdorff global: pick end with fewer options
                if len(head_moves) < len(tail_moves):
                    choose_head = True
                elif len(tail_moves) < len(head_moves):
                    choose_head = False
                else:
                    choose_head = rng.random() < 0.5

            if choose_head:
                # Pick move with min sub_count
                min_sub = min(m[0] for m in head_moves)
                candidates = [m for m in head_moves if m[0] == min_sub]
                _, b, v = rng.choice(candidates)
                path.insert(0, v)
                used_set.add(v)
            else:
                min_sub = min(m[0] for m in tail_moves)
                candidates = [m for m in tail_moves if m[0] == min_sub]
                _, b, v = rng.choice(candidates)
                path.append(v)
                used_set.add(v)
            
            steps += 1
            
        return path

    def local_insert_search(path, used_set, rng, max_iters=50):
        """
        Try to insert vertices into gaps in the path to extend it.
        Uses systematic gap scanning plus random sampling for better coverage.
        """
        current_path = path[:]
        current_used = used_set.copy()
        improved = False
        
        def try_insert_at_pos(pos):
            """Try to insert a vertex at gap pos. Returns True if successful."""
            if pos < 1 or pos >= len(current_path):
                return False
            a = current_path[pos - 1]
            c = current_path[pos]
            
            # Find v such that v~a, v~c, v not in used, v not adj to others
            candidates = []
            for b in range(N):
                v = a ^ (1 << b)
                if v == c or v in current_used:
                    continue
                if (v ^ c).bit_count() != 1:
                    continue
                # Check validity against rest of path
                valid = True
                for i, p in enumerate(current_path):
                    if i == pos - 1 or i == pos:
                        continue
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            
            if candidates:
                v = rng.choice(candidates)
                current_path.insert(pos, v)
                current_used.add(v)
                return True
            return False
        
        # Phase 1: Systematic sweep through all gaps
        for sweep in range(min(5, max_iters)):
            if time.time() > deadline:
                break
            for pos in range(1, len(current_path)):
                if try_insert_at_pos(pos):
                    improved = True
                    break  # Restart sweep since indices shifted
            else:
                break  # No more insertions possible in this sweep
        
        # Phase 2: Random gap sampling for remaining budget
        remaining = max_iters - sweep - 1
        for _ in range(max(0, remaining)):
            if time.time() > deadline:
                break
            if len(current_path) < 2:
                break
            pos = rng.randint(1, len(current_path) - 1)
            if try_insert_at_pos(pos):
                improved = True
        
        return current_path if improved else path

    def final_polish(path, used_set, deadline):
        """
        Exhaustive local search: repeatedly try all possible insertions
        until no more can be made or time runs out.
        """
        current_path = path[:]
        current_used = used_set.copy()
        
        while True:
            if time.time() > deadline:
                break
            if len(current_path) < 2:
                break
            
            inserted = False
            for pos in range(1, len(current_path)):
                if time.time() > deadline:
                    break
                a = current_path[pos - 1]
                c = current_path[pos]
                
                candidates = []
                for b in range(N):
                    v = a ^ (1 << b)
                    if v == c or v in current_used:
                        continue
                    if (v ^ c).bit_count() != 1:
                        continue
                    valid = True
                    for i, p in enumerate(current_path):
                        if i == pos - 1 or i == pos:
                            continue
                        if (v ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        candidates.append(v)
                
                if candidates:
                    v = candidates[0]  # Deterministic pick for polish
                    current_path.insert(pos, v)
                    current_used.add(v)
                    inserted = True
                    break  # Restart from beginning
            
            if not inserted:
                break  # No more insertions possible
        
        return current_path

    def run_one_attempt(start_v, rng):
        """Execute one full construction attempt."""
        path = [start_v]
        used = {start_v}
        
        # Add a few random steps to break symmetry
        for _ in range(3):
            candidates = []
            for b in range(N):
                v = path[-1] ^ (1 << b)
                if is_valid_tail(v, path, used):
                    candidates.append(v)
            if not candidates:
                break
            v = rng.choice(candidates)
            path.append(v)
            used.add(v)
            
        # Bidirectional Warnsdorff Growth
        path = warnsdorff_grow(path, used, rng)
        
        # Local Insertion Search (more iterations)
        path = local_insert_search(path, used, rng, max_iters=50)
        
        return path

    # Main Loop: Multiple Restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        seed = restarts * 104729 + 7
        rng = random.Random(seed)
        
        # Vary starting vertex to explore different regions
        start_v = rng.randint(0, MAX_VERTS - 1)
        
        path = run_one_attempt(start_v, rng)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        # Check if we should stop growing restarts and start polishing
        remaining = deadline - time.time()
        if best_len >= 150 and remaining > 5.0:
            # Spend remaining time on exhaustive polish of best path
            break

    # Final polish phase: exhaustively try insertions on the best path
    if best_path is not None and time.time() < deadline - 2.0:
        best_used = set(best_path)
        polished = final_polish(best_path, best_used, deadline - 1.0)
        if len(polished) > best_len:
            best_path = polished
            best_len = len(best_path)

    if best_path is None:
        # Fallback: Simple greedy if something went wrong
        path = [0]
        used = {0}
        while True:
            found = False
            for b in range(N):
                v = path[-1] ^ (1 << b)
                if is_valid_tail(v, path, used):
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        return path

    return best_path

# EVOLVE-BLOCK-END
