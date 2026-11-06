import time
import random
import sys
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
# Safety margin: Evaluator kills at 90s. We stop at 85s to allow for cleanup/return.
DEADLINE_SECONDS = 85.0 

def _build_subset_trans(trans):
    """Precompute subset transitions for one letter using dynamic programming."""
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    Uses BFS on the subset automaton with precomputed transitions.
    """
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)

    # BFS from FULL_MASK
    size = 1 << N
    # Using a list of ints is faster than bytearray for random access in Python for this size
    # but bytearray is more memory efficient. Let's stick to a list of ints for speed, 
    # initialized to -1.
    dist = [-1] * size 
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])

    while queue:
        cur = queue.popleft()

        # Check if singleton
        if cur & (cur - 1) == 0:
            return dist[cur]

        d = dist[cur] + 1
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = d
            queue.append(nxt_a)

        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = d
            queue.append(nxt_b)

    return 0

def _generate_initial_candidates():
    """
    Generate a set of high-quality starting points for the search.
    Ping-Pong automata are known to have high thresholds.
    """
    candidates = []
    
    # 1. Standard Ping-Pong: a=(0 1 ... N-1), b=(0 N-1 ... 1) is not quite right for binary.
    # Classic Ping-Pong: a shifts +1, b shifts -1 (or similar inverse permutation)
    # Actually, the "Ping-Pong" automaton usually has a as a cycle and b as a "reversal" or similar.
    # Let's use the specific structure that often yields high scores:
    # a: 0->1, 1->2, ..., N-1->0 (Cycle)
    # b: 0->0, 1->0, ..., N-1->0 (Reset) -> This has low threshold.
    
    # High threshold structures often involve two permutations that don't commute easily.
    # Example: a = (0 1 2 ... 11), b = (0 2 4 ... 10 1 3 ... 11) ? No.
    
    # Let's try the "Krotov" or "Pinwheel" type structures or just random permutations.
    # A known high-threshold 12-state automaton might be found by perturbing a cycle.
    
    # Candidate 1: a is cycle, b is a different cycle
    a_cycle = [(i + 1) % N for i in range(N)]
    for shift in range(1, N):
        if shift == 1: continue
        b_cycle = [(i + shift) % N for i in range(N)]
        candidates.append((a_cycle[:], b_cycle[:]))
        
    # Candidate 2: a is cycle, b has a "defect"
    for defect_pos in range(N):
        for defect_dest in range(N):
            if defect_dest == (defect_pos + 1) % N: continue
            a = a_cycle[:]
            b = a_cycle[:] # Start with same cycle, then break it
            b[defect_pos] = defect_dest
            candidates.append((a[:], b[:]))
            
    # Candidate 3: Random permutations (often good starting points for SA)
    for _ in range(10):
        a = list(range(N))
        random.shuffle(a)
        b = list(range(N))
        random.shuffle(b)
        candidates.append((a[:], b[:]))
        
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Set seed for reproducibility if needed, but random is fine for SA
    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0
    
    # Phase 1: Evaluate initial candidates to find a good starting point
    candidates = _generate_initial_candidates()
    
    # Shuffle to avoid bias
    random.shuffle(candidates)
    
    for a, b in candidates:
        if time.time() > deadline - 2.0:
            break
        score = compute_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            
    # If we didn't find a synchronizing automaton, we must ensure we return a valid one.
    # However, the task requires maximizing the score of a synchronizing automaton.
    # If best_score is 0, we need a fallback.
    if best_score == 0:
         # Fallback: A simple synchronizing automaton
         # a: cycle, b: reset to 0
         a = [(i + 1) % N for i in range(N)]
         b = [0] * N
         best_a = a
         best_b = b
         best_score = compute_threshold(a, b)
         
    if best_a is None:
         # Absolute fallback
         a = [(i + 1) % N for i in range(N)]
         b = [0] * N
         return [a, b]

    # Phase 2: Simulated Annealing
    # We start from the best candidate found so far.
    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_score = best_score
    
    # SA Parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    # We don't know how many steps we can take, so we loop until time is up.
    
    # To speed up, we can precompute subset transitions for current a and b?
    # But they change. So we recompute in compute_threshold.
    # This is the bottleneck.
    
    # Optimization: We can keep the next_subset arrays if we only change one transition.
    # But implementing an incremental update for the subset DP is complex.
    # Given N=12, _build_subset_trans is O(2^N * N) approx? No, it's O(2^N).
    # 4096 iterations. Very fast in Python if optimized.
    # compute_threshold is BFS on 4096 nodes. Also fast.
    # Let's profile mentally: 
    # _build_subset_trans: 4096 loops.
    # BFS: 4096 nodes, 2 edges each. ~8000 ops.
    # Total per eval: ~12k ops.
    # Python can do ~10M-50M simple ops/sec.
    # So ~1000-4000 evals/sec.
    # 85 seconds * 1000 = 85,000 steps. Plenty for SA.
    
    while time.time() < deadline - 1.0:
        # Generate a move
        # Change one destination in a or b
        if random.random() < 0.5:
            state = random.randint(0, N - 1)
            old_dest = cur_a[state]
            new_dest = random.randint(0, N - 1)
            if new_dest == old_dest:
                continue
            cur_a[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            
            if new_score == 0:
                # Not synchronizing, revert
                cur_a[state] = old_dest
                continue
            # SA acceptance
            delta = new_score - cur_score
            if delta > 0 or random.random() < math.exp(delta / T):
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_a[state] = old_dest
        else:
            state = random.randint(0, N - 1)
            old_dest = cur_b[state]
            new_dest = random.randint(0, N - 1)
            if new_dest == old_dest:
                continue
            cur_b[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            
            if new_score == 0:
                cur_b[state] = old_dest
                continue
            delta = new_score - cur_score
            if delta > 0 or random.random() < math.exp(delta / T):
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_b[state] = old_dest
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Restart SA with higher T if stuck?
            # For now, just keep going.
            
    return [best_a, best_b]

if __name__ == "__main__":
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"A: {a}")
    print(f"B: {b}")

