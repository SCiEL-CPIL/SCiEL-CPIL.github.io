import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _precompute_transitions(trans):
    """Precompute subset transitions for both letters using bitmask operations."""
    # For each letter, precompute the image of each state as a bitmask
    img0 = [0] * N
    img1 = [0] * N
    for i in range(N):
        img0[i] = 1 << trans[0][i]
        img1[i] = 1 << trans[1][i]
    
    # Precompute subset transitions for all 2^N subsets
    # trans_table[letter][mask] = resulting mask
    trans_table = [[0] * (1 << N) for _ in range(2)]
    for mask in range(1 << N):
        m0 = 0
        m1 = 0
        for i in range(N):
            if (mask >> i) & 1:
                m0 |= img0[i]
                m1 |= img1[i]
        trans_table[0][mask] = m0
        trans_table[1][mask] = m1
    return trans_table

def compute_threshold(trans):
    """
    Compute the shortest reset word length using BFS on the subset automaton.
    trans: list of two lists, each of length N.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    trans_table = _precompute_transitions(trans)
    start = FULL_MASK
    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current.bit_count() == 1:
            return dist
            
        for letter in range(2):
            next_mask = trans_table[letter][current]
            if next_mask not in visited:
                visited[next_mask] = dist + 1
                queue.append(next_mask)
                
    return -1

def _fast_threshold(trans):
    """Optimized threshold computation with early exit."""
    trans_table = _precompute_transitions(trans)
    start = FULL_MASK
    visited = set()
    visited.add(start)
    queue = deque([(start, 0)])
    
    while queue:
        current, dist = queue.popleft()
        
        if current.bit_count() == 1:
            return dist
            
        for letter in range(2):
            next_mask = trans_table[letter][current]
            if next_mask not in visited:
                visited.add(next_mask)
                queue.append((next_mask, dist + 1))
                
    return -1

def _generate_ping_pong_structures():
    """Generate ping-pong style automata with potentially high reset thresholds."""
    structures = []
    
    # Type 1: Cycle + collapse to single state
    for start_state in range(N):
        trans_a = [(i + 1) % N for i in range(N)]
        trans_b = [start_state] * N
        structures.append([trans_a[:], trans_b[:]])
    
    # Type 2: Cycle + collapse to two states
    for s1 in range(N):
        for s2 in range(N):
            if s2 != s1:
                trans_a = [(i + 1) % N for i in range(N)]
                trans_b = [s1 if i % 2 == 0 else s2 for i in range(N)]
                structures.append([trans_a[:], trans_b[:]])
    
    # Type 3: Two partial cycles with different splits
    for split in range(1, N):
        trans_a = []
        trans_b = []
        for i in range(N):
            if i < split:
                trans_a.append((i + 1) % split)
                trans_b.append((i + 1) % N)
            else:
                trans_a.append((i + 1) % N)
                trans_b.append((i - split + 1) % (N - split))
        structures.append([trans_a[:], trans_b[:]])
    
    # Type 4: Randomized ping-pong
    for _ in range(100):
        cycle_len = random.randint(3, N)
        trans_a = [(i + 1) % cycle_len if i < cycle_len else random.randint(0, cycle_len - 1) for i in range(N)]
        collapse_target = random.randint(0, N - 1)
        trans_b = [collapse_target if random.random() < 0.7 else random.randint(0, N - 1) for i in range(N)]
        structures.append([trans_a[:], trans_b[:]])
    
    # Type 5: Near-cycle structures
    for _ in range(50):
        trans_a = [(i + 1) % N for i in range(N)]
        # Perturb one or two transitions
        for _ in range(random.randint(1, 2)):
            trans_a[random.randint(0, N - 1)] = random.randint(0, N - 1)
        trans_b = [random.randint(0, N - 1) for _ in range(N)]
        structures.append([trans_a[:], trans_b[:]])
    
    return structures

def _refine_automaton(trans, deadline, verbose=False):
    """
    Refine an automaton using simulated annealing to maximize reset threshold.
    Uses precomputed transition tables for speed.
    """
    current = [row[:] for row in trans]
    current_thresh = _fast_threshold(current)
    if current_thresh < 0:
        return None, -1
        
    best = [row[:] for row in current]
    best_thresh = current_thresh
    
    # Precompute for current
    current_table = _precompute_transitions(current)
    
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.01
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline and T > min_T:
        # Mutate: change one transition
        new_trans = [row[:] for row in current]
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        if new_dest == current[letter][state]:
            continue
        new_trans[letter][state] = new_dest
        
        new_thresh = _fast_threshold(new_trans)
        
        if new_thresh > current_thresh:
            current = new_trans
            current_thresh = new_thresh
            current_table = _precompute_transitions(current)
            if current_thresh > best_thresh:
                best = [row[:] for row in current]
                best_thresh = current_thresh
                no_improve_count = 0
            T = max(min_T, T * 1.1)
        else:
            delta = new_thresh - current_thresh
            if delta < 0:
                if T > 0.01:
                    prob = 2.718281828 ** (delta / T)
                    if random.random() < prob:
                        current = new_trans
                        current_thresh = new_thresh
                        current_table = _precompute_transitions(current)
            elif delta == 0:
                if random.random() < 0.2:
                    current = new_trans
                    current_table = _precompute_transitions(current)
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart with higher temperature
                T = 5.0
                no_improve_count = 0
                # Random restart from best
                current = [row[:] for row in best]
                current_thresh = best_thresh
                current_table = _precompute_transitions(current)
    
    return best, best_thresh

def _local_search(trans, deadline):
    """
    Local search: try all single-transition changes and pick the best.
    Repeat until no improvement or time runs out.
    """
    current = [row[:] for row in trans]
    current_thresh = _fast_threshold(current)
    if current_thresh < 0:
        return None, -1
        
    best = [row[:] for row in current]
    best_thresh = current_thresh
    
    while time.time() < deadline:
        improved = False
        for letter in range(2):
            for state in range(N):
                for new_dest in range(N):
                    if new_dest == current[letter][state]:
                        continue
                    new_trans = [row[:] for row in current]
                    new_trans[letter][state] = new_dest
                    new_thresh = _fast_threshold(new_trans)
                    if new_thresh > current_thresh:
                        current = new_trans
                        current_thresh = new_thresh
                        improved = True
                        if current_thresh > best_thresh:
                            best = [row[:] for row in current]
                            best_thresh = current_thresh
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_thresh

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    best_trans = None
    best_threshold = 0
    
    # Phase 1: Try structured ping-pong automata
    structures = _generate_ping_pong_structures()
    
    for struct in structures:
        if time.time() > deadline - 20:
            break
        thresh = _fast_threshold(struct)
        if thresh > best_threshold:
            best_threshold = thresh
            best_trans = [row[:] for row in struct]
    
    # Phase 2: Local search from best structured automaton
    if best_trans is not None and time.time() < deadline - 15:
        ls_result, ls_thresh = _local_search(best_trans, deadline - 15)
        if ls_result is not None and ls_thresh > best_threshold:
            best_threshold = ls_thresh
            best_trans = ls_result
    
    # Phase 3: Simulated annealing from best
    if best_trans is not None and time.time() < deadline - 10:
        sa_result, sa_thresh = _refine_automaton(best_trans, deadline - 10)
        if sa_result is not None and sa_thresh > best_threshold:
            best_threshold = sa_thresh
            best_trans = sa_result
    
    # Phase 4: More SA restarts from different initial points
    remaining = deadline - time.time()
    if remaining > 10:
        num_restarts = min(5, int(remaining / 3))
        for _ in range(num_restarts):
            if time.time() > deadline - 3:
                break
            # Start from a perturbed version of best
            if best_trans is not None:
                init_trans = [row[:] for row in best_trans]
                for _ in range(random.randint(3, 8)):
                    letter = random.randint(0, 1)
                    state = random.randint(0, N - 1)
                    init_trans[letter][state] = random.randint(0, N - 1)
            else:
                init_trans = [
                    [random.randint(0, N - 1) for _ in range(N)],
                    [random.randint(0, N - 1) for _ in range(N)]
                ]
            
            chunk_deadline = min(deadline - 2, time.time() + remaining / (num_restarts - _))
            sa_result, sa_thresh = _refine_automaton(init_trans, chunk_deadline)
            if sa_result is not None and sa_thresh > best_threshold:
                best_threshold = sa_thresh
                best_trans = sa_result
    
    # Final: local search polish
    if best_trans is not None and time.time() < deadline - 1:
        ls_result, ls_thresh = _local_search(best_trans, deadline - 1)
        if ls_result is not None and ls_thresh > best_threshold:
            best_threshold = ls_thresh
            best_trans = ls_result
    
    if best_trans is None:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return best_trans
