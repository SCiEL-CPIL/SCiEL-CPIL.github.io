import time
import random
from collections import deque

N = 12

def compute_threshold(trans):
    """
    Compute the shortest reset word length using BFS on the subset automaton.
    trans: list of two lists, each of length N.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    start = full_mask
    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current.bit_count() == 1:
            return dist
            
        # Letter 0
        next_mask = 0
        for i in range(N):
            if (current >> i) & 1:
                next_mask |= 1 << trans[0][i]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
        # Letter 1
        next_mask = 0
        for i in range(N):
            if (current >> i) & 1:
                next_mask |= 1 << trans[1][i]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
    return -1

def _generate_ping_pong_structures():
    """
    Generate ping-pong style automata with high reset thresholds.
    Ping-pong structure: one letter is a cycle, the other collapses to a subset.
    """
    structures = []
    
    # Type 1: Cycle + collapse to single state
    for start_state in range(N):
        trans_a = [(i + 1) % N for i in range(N)]  # Full cycle
        trans_b = [start_state] * N  # Collapse to one state
        structures.append([trans_a[:], trans_b[:]])
    
    # Type 2: Cycle + collapse to two states
    for start_state in range(N):
        for second_state in range(N):
            if second_state != start_state:
                trans_a = [(i + 1) % N for i in range(N)]
                trans_b = []
                for i in range(N):
                    if i % 2 == 0:
                        trans_b.append(start_state)
                    else:
                        trans_b.append(second_state)
                structures.append([trans_a[:], trans_b[:]])
    
    # Type 3: Two partial cycles
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
    for _ in range(50):
        cycle_len = random.randint(3, N)
        trans_a = []
        for i in range(N):
            if i < cycle_len:
                trans_a.append((i + 1) % cycle_len)
            else:
                trans_a.append(random.randint(0, cycle_len - 1))
        
        collapse_target = random.randint(0, N - 1)
        trans_b = []
        for i in range(N):
            if random.random() < 0.7:
                trans_b.append(collapse_target)
            else:
                trans_b.append(random.randint(0, N - 1))
        structures.append([trans_a[:], trans_b[:]])
    
    return structures

def _refine_automaton(trans, deadline):
    """
    Refine an automaton using simulated annealing to maximize reset threshold.
    """
    current = [row[:] for row in trans]
    current_thresh = compute_threshold(current)
    best = [row[:] for row in current]
    best_thresh = current_thresh
    
    T = 5.0
    cooling_rate = 0.99
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        # Mutate: change one transition
        new_trans = [row[:] for row in current]
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        new_trans[letter][state] = new_dest
        
        new_thresh = compute_threshold(new_trans)
        
        if new_thresh > current_thresh:
            current = new_trans
            current_thresh = new_thresh
            if current_thresh > best_thresh:
                best = [row[:] for row in current]
                best_thresh = current_thresh
            T = max(min_T, T * 1.05)  # Heat up slightly
        else:
            delta = new_thresh - current_thresh
            if delta < 0:
                prob = 2.718281828 ** (delta / T) if T > 0.01 else 0
                if random.random() < prob:
                    current = new_trans
                    current_thresh = new_thresh
            elif delta == 0:
                if random.random() < 0.3:
                    current = new_trans
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
    
    return best, best_thresh

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    best_trans = None
    best_threshold = 0
    
    # Phase 1: Try structured ping-pong automata
    structures = _generate_ping_pong_structures()
    
    for struct in structures:
        if time.time() > deadline - 10:
            break
        thresh = compute_threshold(struct)
        if thresh > best_threshold:
            best_threshold = thresh
            best_trans = [row[:] for row in struct]
    
    # Phase 2: Refine the best structured automaton
    if best_trans is not None:
        refined, refined_thresh = _refine_automaton(best_trans, deadline - 5)
        if refined_thresh > best_threshold:
            best_threshold = refined_thresh
            best_trans = refined
    else:
        # Fallback: try a simple synchronizing automaton
        best_trans = [[(i + 1) % N for i in range(N)], [0] * N]
        best_threshold = compute_threshold(best_trans)
    
    # Phase 3: Additional refinement with different initial conditions
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Try a few more random starts
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            random_trans = [
                [random.randint(0, N - 1) for _ in range(N)],
                [random.randint(0, N - 1) for _ in range(N)]
            ]
            refined, refined_thresh = _refine_automaton(random_trans, deadline - 2)
            if refined_thresh > best_threshold:
                best_threshold = refined_thresh
                best_trans = refined
    
    if best_trans is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return best_trans
