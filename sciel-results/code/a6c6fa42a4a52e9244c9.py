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
        """
        current_path = path[:]
        current_used = used_set.copy()
        improved = False
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            if len(current_path) < 2:
                break
                
            # Pick a random gap
            pos = rng.randint(1, len(current_path) - 1)
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
                # Insert and verify (verification is cheap enough for small N if we just check the insertion point locally? 
                # Actually, we checked against all others, so it's valid. But let's be safe and just update)
                current_path.insert(pos, v)
                current_used.add(v)
                improved = True
        
        return current_path if improved else path

    def run_one_attempt(start_v, rng):
        """Execute one full construction attempt."""
        path = [start_v]
        used = {start_v}
        
        # Initial seed: just 1 vertex is fine, Warnsdorff handles the rest.
        # Or add a few random steps to break symmetry?
        # Let's add 2 random steps to give it a direction.
        for _ in range(2):
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
        
        # Local Insertion Search
        path = local_insert_search(path, used, rng, max_iters=20)
        
        return path

    # Main Loop: Multiple Restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        seed = restarts * 104729 + 7
        rng = random.Random(seed)
        
        # Vary starting vertex slightly
        # 0 is a good start, but maybe try others?
        # In Q_N, all vertices are symmetric, so 0 is fine.
        # But maybe random start helps escape specific local minima if the heuristic is deterministic-ish?
        # Warnsdorff is deterministic given the state, but we have random tie-breaking.
        start_v = 0 
        
        path = run_one_attempt(start_v, rng)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        # If we've hit a very high score, we might want to stop early?
        # Theoretical max is unknown but likely < 512.
        # If we hit 200+, we are doing well.
        if best_len > 190:
             # We can reduce max_iters in local search or just continue.
             pass

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
