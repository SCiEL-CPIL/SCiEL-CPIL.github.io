import time
import random
import sys
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 85.0  # safety margin before 90s kill


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


def _generate_ping_pong_structures():
    """
    Generate 'near-perfect' ping-pong structures.
    Strategy: One letter is a full cycle. The other is a cycle with 1 or 2 defects.
    This focuses the search on the known high-threshold family.
    """
    structures = []
    cycle_a = [(i + 1) % N for i in range(N)]
    
    # Case 1: 'a' is full cycle, 'b' has 1 defect
    for defect_pos in range(N):
        for defect_dest in range(N):
            if defect_dest == (defect_pos + 1) % N:
                continue
            b = cycle_a[:]
            b[defect_pos] = defect_dest
            structures.append((cycle_a[:], b[:]))
            
    # Case 2: 'a' is full cycle, 'b' has 2 defects
    # We limit the number of pairs to avoid explosion, but N=12 is small enough
    # to check all pairs if we are efficient, but let's sample heavily.
    # Actually, let's just generate all pairs of defects for the initial set.
    # 12*11/2 = 66 pairs. For each pair, 11*11 dests. ~8000 structures.
    # This is fast to evaluate.
    for pos1 in range(N):
        for pos2 in range(pos1 + 1, N):
            for dest1 in range(N):
                if dest1 == (pos1 + 1) % N:
                    continue
                for dest2 in range(N):
                    if dest2 == (pos2 + 1) % N:
                        continue
                    b = cycle_a[:]
                    b[pos1] = dest1
                    b[pos2] = dest2
                    structures.append((cycle_a[:], b[:]))
                    
    # Also consider the reverse: 'b' is full cycle, 'a' has defects
    # Due to symmetry, this is equivalent to swapping a and b in the above,
    # but the threshold calculation is symmetric so we don't need to duplicate
    # the search space if we treat (a,b) and (b,a) as same.
    # However, local search might get stuck, so having both orientations in pool is good.
    for defect_pos in range(N):
        for defect_dest in range(N):
            if defect_dest == (defect_pos + 1) % N:
                continue
            a = cycle_a[:]
            a[defect_pos] = defect_dest
            structures.append((a[:], cycle_a[:]))
            
    for pos1 in range(N):
        for pos2 in range(pos1 + 1, N):
            for dest1 in range(N):
                if dest1 == (pos1 + 1) % N:
                    continue
                for dest2 in range(N):
                    if dest2 == (pos2 + 1) % N:
                        continue
                    a = cycle_a[:]
                    a[pos1] = dest1
                    a[pos2] = dest2
                    structures.append((a[:], cycle_a[:]))

    return structures


def _is_cycle(trans):
    """Check if a transition function is a single cycle of length N."""
    visited = [False] * N
    for i in range(N):
        if not visited[i]:
            count = 0
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cur = trans[cur]
                count += 1
            if count != N:
                return False
    return True

def _refine_automaton(a, b, start_time, deadline):
    """
    Targeted local search.
    Strategy: Identify which letter is 'more cyclic'. 
    Bias mutations to preserve the cycle structure of the 'anchor' letter 
    and modify the 'defect' letter.
    """
    cur_a = a[:]
    cur_b = b[:]
    cur_score = compute_threshold(cur_a, cur_b)

    if cur_score == 0:
        return None, None, 0

    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score

    # Determine anchor: which letter is closer to a full cycle?
    # We count how many states i satisfy trans[i] == (i+1)%N
    def count_cycle_preserving(trans):
        return sum(1 for i in range(N) if trans[i] == (i + 1) % N)
    
    a_pres = count_cycle_preserving(cur_a)
    b_pres = count_cycle_preserving(cur_b)
    
    # If one is a perfect cycle, it's the anchor.
    # If not, the one with higher preservation is the anchor.
    anchor_is_a = a_pres >= b_pres

    no_improve = 0
    max_no_improve = 100
    
    # Precompute next subsets for speed? 
    # compute_threshold already does this. 
    # To speed up local search, we could maintain the subset automaton, 
    # but rebuilding is O(2^N) which is ~4096 ops, very fast.
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
            
        # 80% chance: modify the non-anchor (defect) letter
        # 20% chance: modify the anchor letter (to escape local optima)
        if random.random() < 0.8:
            if anchor_is_a:
                target_trans = cur_b
                is_a = False
            else:
                target_trans = cur_a
                is_a = True
        else:
            if anchor_is_a:
                target_trans = cur_a
                is_a = True
            else:
                target_trans = cur_b
                is_a = False

        state = random.randint(0, N - 1)
        old_dest = target_trans[state]
        
        # Smart mutation: 
        # 1. If target is anchor, try to keep it cyclic (swap with neighbor?)
        # 2. If target is defect, try random dest or swap with another state's dest
        
        new_dest = random.randint(0, N - 1)
        if new_dest == old_dest:
            continue
            
        # Apply mutation
        if is_a:
            cur_a[state] = new_dest
        else:
            cur_b[state] = new_dest
            
        new_score = compute_threshold(cur_a, cur_b)
        
        if new_score > cur_score:
            cur_score = new_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            # Revert
            if is_a:
                cur_a[state] = old_dest
            else:
                cur_b[state] = old_dest
            no_improve += 1

    return best_a, best_b, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0

    # Phase 1: Evaluate structured candidates (Ping-Pong family)
    structures = _generate_ping_pong_structures()
    # Shuffle to ensure we don't bias towards the first found, 
    # though we evaluate all if time permits.
    random.shuffle(structures)

    for a_struct, b_struct in structures:
        if time.time() > deadline - 10.0:
            break

        score = compute_threshold(a_struct, b_struct)
        if score > best_score:
            best_score = score
            best_a = a_struct[:]
            best_b = b_struct[:]

    # Phase 2: Local search refinement from best structured candidate
    if best_a is not None and time.time() < deadline - 5.0:
        refined_a, refined_b, refined_score = _refine_automaton(
            best_a, best_b, start_time, deadline - 5.0
        )
        if refined_score > best_score:
            best_score = refined_score
            best_a = refined_a
            best_b = refined_b

    # Phase 3: Try a few random starts with refinement 
    # to escape potential global local optima of the ping-pong family
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        num_random_starts = min(5, int(remaining_time / 4))
        for _ in range(num_random_starts):
            if time.time() > deadline - 3.0:
                break

            a_rand = [random.randint(0, N - 1) for _ in range(N)]
            b_rand = [random.randint(0, N - 1) for _ in range(N)]

            score = compute_threshold(a_rand, b_rand)
            if score == 0:
                continue

            if score > best_score:
                best_score = score
                best_a = a_rand[:]
                best_b = b_rand[:]

            # Only refine if it's competitive
            if score >= best_score - 5 and time.time() < deadline - 3.0:
                refined_a, refined_b, refined_score = _refine_automaton(
                    a_rand, b_rand, start_time, deadline - 3.0
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_a = refined_a
                    best_b = refined_b

    # Fallback: ensure we return a valid synchronizing automaton
    if best_score == 0 or best_a is None:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]

    return [best_a, best_b]


if __name__ == "__main__":
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"A: {a}")
    print(f"B: {b}")
