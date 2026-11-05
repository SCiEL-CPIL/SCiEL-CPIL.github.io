"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
TIME_LIMIT = 110  # safety margin below 120s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_path = None
    best_len = 0

    def elapsed():
        return time.time() - start_time

    def time_left():
        return TIME_LIMIT - elapsed()

    def get_neighbors(v):
        """Generate all neighbors of v in Q_N."""
        for b in range(N):
            yield v ^ (1 << b)

    def extend_path_greedy(path, used, forbidden, bit_order, max_steps=None):
        """Extend path greedily. Maintains 'forbidden' set of neighbors of non-last vertices."""
        if max_steps is None:
            max_steps = 512
        for _ in range(max_steps):
            cur = path[-1]
            found = False
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    # Accept v
                    # Update forbidden: neighbors of cur (old last) become forbidden
                    for nb in get_neighbors(cur):
                        forbidden.add(nb)
                    
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        return path

    def try_greedy_with_lookahead(start_vertex, bit_order, max_steps=512):
        """Greedy with simple lookahead using forbidden set."""
        path = [start_vertex]
        used = {start_vertex}
        forbidden = set()
        
        # Initialize forbidden with neighbors of start_vertex? 
        # No, start_vertex is the only vertex, so it is the last one.
        # Its neighbors are NOT forbidden yet.
        
        for _ in range(max_steps):
            cur = path[-1]
            best_candidate = None
            best_score = -1

            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used or v in forbidden:
                    continue
                
                # Score: count valid extensions from v
                score = 0
                for b2 in bit_order:
                    v2 = v ^ (1 << b2)
                    if v2 == cur:
                        continue
                    # Check if v2 is valid for a path ending in v
                    # v2 must not be in used
                    # v2 must not be in forbidden (neighbors of path[:-1])
                    # v2 must not be neighbor of v? No, v2 is neighbor of v by definition.
                    # The condition for v2 being a valid NEXT step from v is:
                    # v2 not in used
                    # v2 not in forbidden
                    # Note: 'forbidden' currently contains neighbors of path[0]...path[-2].
                    # If we were to append v, the new forbidden set would include neighbors of cur.
                    # But for the lookahead score, we just want to know if v has many options.
                    # We check if v2 is free.
                    if v2 not in used and v2 not in forbidden:
                        # Also v2 should not be adjacent to v? It is.
                        # Is v2 adjacent to any other path vertex?
                        # If v2 is in forbidden, it is adjacent to some path[i] i < -1.
                        # So checking 'not in forbidden' is sufficient for the non-consecutive constraint.
                        # We don't need to check adjacency to v because v2 is a neighbor of v.
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_candidate = v

            if best_candidate is None:
                break

            # Commit to best_candidate
            # Update forbidden: neighbors of cur become forbidden
            for nb in get_neighbors(cur):
                forbidden.add(nb)
                
            path.append(best_candidate)
            used.add(best_candidate)

        return path

    def random_bit_order():
        order = list(range(N))
        random.shuffle(order)
        return order

    best_path = None
    best_len = 0

    # Helper to run a single greedy attempt
    def run_attempt(sv, bo):
        path = [sv]
        used = {sv}
        forbidden = set()
        # Initial forbidden is empty because sv is the last vertex.
        return extend_path_greedy(path, used, forbidden, bo)

    # Try baseline
    p = run_attempt(0, list(range(N)))
    best_path = p
    best_len = len(p)

    # Random restarts
    while time_left() > 2.0:
        sv = random.randint(0, 2**N - 1)
        bo = random_bit_order()
        p = run_attempt(sv, bo)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    return best_path
# EVOLVE-BLOCK-END
