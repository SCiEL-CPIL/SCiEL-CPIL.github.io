"""Combined program: Warnsdorff heuristic with backtracking and efficient exclusion sets."""
import time
import random
import sys

N = 9
TIME_LIMIT = 118.0  # Safety margin below 120s evaluator limit
SEED_LENGTH = 104

# Pre-compute neighbors for all vertices in Q_9
# This avoids bit operations in the inner loop
NEIGHBORS = [[] for _ in range(1 << N)]
for v in range(1 << N):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# EVOLVE-BLOCK-START
def _count_valid_next(v, used, exclusion):
    """
    Count how many valid moves are available from vertex v,
    given the current used set and exclusion set.
    A move to w is valid if w is not used and not in exclusion.
    """
    count = 0
    for w in NEIGHBORS[v]:
        if w not in used and w not in exclusion:
            count += 1
    return count

def _warnsdorff_dfs(path, used, exclusion, rng, time_budget, backtrack_limit=5):
    """
    Perform a DFS with Warnsdorff heuristic and limited backtracking.
    Returns the longest path found.
    """
    best_path = list(path)
    best_len = len(path)
    
    # Stack for backtracking: stores (path_snapshot, used_snapshot, exclusion_snapshot, backtrack_remaining)
    # We don't want to copy huge sets every time, so we use a simpler approach:
    # We will maintain the state and only backtrack by undoing the last move.
    
    # Since backtracking in a set-based state is tricky to undo efficiently without copies,
    # and the path length is small (< 200), we can afford to copy sets if we limit the depth of backtracking.
    # However, a better approach is to use a recursive function with explicit state management.
    
    # Let's use a recursive helper that takes current state and returns best extension.
    # To avoid deep recursion limits and overhead, we'll implement a manual stack or limited recursion.
    
    # Given the constraints, a simple recursive DFS with a depth limit for backtracking is sufficient.
    # We will track the "depth" of backtracking allowed.
    
    def dfs(current_path, current_used, current_exclusion, bt_left):
        nonlocal best_path, best_len
        
        if time.time() > time_budget:
            return
        
        if len(current_path) > best_len:
            best_len = len(current_path)
            best_path = list(current_path)
            
        cur = current_path[-1]
        
        # Generate candidates and score them using Warnsdorff
        candidates = []
        for v in NEIGHBORS[cur]:
            if v in current_used or v in current_exclusion:
                continue
            # Warnsdorff score: number of valid next moves from v
            # Note: We must check validity against current_used and current_exclusion.
            # The 'exclusion' set for the new state would include neighbors of cur.
            # But for the heuristic, we just want to know how "tight" v is.
            # A vertex v is "tight" if it has few valid neighbors in the *current* state.
            # However, when we move to v, the old terminal cur becomes non-terminal,
            # so all neighbors of cur are added to exclusion.
            # This might invalidate some of v's neighbors if they are neighbors of cur.
            # But v is a neighbor of cur, so v's neighbors are at dist 1 from v.
            # If a neighbor w of v is also a neighbor of cur, then w is at dist 2 from v? No.
            # If v = cur ^ (1<<b), and w = v ^ (1<<c).
            # If c != b, w = cur ^ (1<<b) ^ (1<<c). dist(w, cur) = 2. So w is not a neighbor of cur.
            # If c == b, w = cur. But cur is in used.
            # So the neighbors of v (other than cur) are NOT neighbors of cur.
            # Therefore, adding cur's neighbors to exclusion does NOT remove any of v's valid neighbors
            # (except cur, which is already used).
            # So the count of valid neighbors from v is a good estimate.
            
            score = _count_valid_next(v, current_used, current_exclusion)
            candidates.append((score, v))
        
        if not candidates:
            return
        
        # Sort by score (fewest exits first)
        candidates.sort(key=lambda x: x[0])
        
        # Add small random noise to break ties and allow variation
        for i in range(len(candidates)):
            candidates[i] = (candidates[i][0] + rng.random() * 0.1, candidates[i][1])
        candidates.sort(key=lambda x: x[0])
        
        for score, v in candidates:
            if time.time() > time_budget:
                return
            
            # Move to v
            new_used = current_used | {v}
            new_exclusion = set(current_exclusion)
            # Add neighbors of cur to exclusion (cur becomes non-terminal)
            for nb in NEIGHBORS[cur]:
                new_exclusion.add(nb)
            
            new_path = current_path + [v]
            
            dfs(new_path, new_used, new_exclusion, bt_left)
            
            # Backtrack
            # We only backtrack if we haven't exhausted our backtracking budget for this branch?
            # Actually, standard DFS backtracks always. The "backtrack_limit" idea was to limit
            # how deep we go into alternative branches after a failure.
            # Here, we simply try all candidates. If we want to limit search, we can stop early.
            # But Warnsdorff usually finds a long path quickly.
            # To prevent explosion, we can limit the number of candidates we try at each step
            # if the path is already quite long.
            
            # Optimization: If we have found a very long path, we can be less exhaustive.
            # But let's try full Warnsdorff first.
            
            # Break after first move if we are confident? No, try all to find the best.
            # However, if we have tried 3 candidates and none improved the global best significantly,
            # we might stop.
            # For now, let's just run DFS.
            
    # Start DFS
    dfs(path, used, exclusion, backtrack_limit)
    
    return best_path

def _fast_greedy(rng, time_budget):
    """
    A fast greedy search without backtracking, used to quickly find a decent baseline
    or to fill time if DFS is slow.
    """
    path = [0]
    used = {0}
    exclusion = set()
    
    while time.time() < time_budget:
        cur = path[-1]
        candidates = []
        for v in NEIGHBORS[cur]:
            if v not in used and v not in exclusion:
                candidates.append(v)
        
        if not candidates:
            break
        
        # Pick random or Warnsdorff? Let's pick random for diversity in this fast mode.
        v = rng.choice(candidates)
        path.append(v)
        used.add(v)
        
        for nb in NEIGHBORS[cur]:
            exclusion.add(nb)
            
    return path

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = [0]
    best_len = 1
    
    sys.setrecursionlimit(10000)
    
    # Initial quick greedy to get a baseline
    rng_init = random.Random(42)
    p_init = _fast_greedy(rng_init, start_time + 2.0)
    if len(p_init) > best_len:
        best_len = len(p_init)
        best_path = p_init
    
    # Main search: Warnsdorff DFS with backtracking
    # We run multiple restarts with different random seeds to explore different parts of the space.
    # The DFS itself is deterministic given the seed and noise, so restarts are needed.
    
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        rng = random.Random(restart_count * 12345 + 999)
        
        # We start from 0, but we could also start from a random vertex if we wanted.
        # Starting from 0 is standard.
        
        # Time budget for this DFS run: give it enough time to go deep.
        # If it hits the deadline, it returns the best found so far.
        # We allocate most of the remaining time to each restart.
        # However, if a restart is slow, it might not finish.
        # Let's give each restart a fixed chunk of time, e.g., 10s, and do many restarts?
        # Or let it run until deadline?
        # Warnsdorff is fast. It usually finishes in < 1s.
        # So we can do many restarts.
        
        run_deadline = min(deadline - 1.0, time.time() + 15.0)
        
        p = _warnsdorff_dfs([0], {0}, set(), rng, run_deadline)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Final cleanup: ensure we return a valid path
    # The DFS ensures validity by construction.
    
    return best_path
# EVOLVE-BLOCK-END
