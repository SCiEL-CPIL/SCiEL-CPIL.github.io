import time
import random
import sys
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
# Safety margin: Evaluator kills at 90s. We aim to finish by 85s.
DEADLINE_SECONDS = 85.0 

def _build_subset_trans(trans):
    """
    Precompute subset transitions for one letter using dynamic programming.
    trans: list of N integers, trans[i] is destination of state i.
    Returns a list of size 2^N where next_subset[mask] is the bitmask of the image of mask under trans.
    """
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        # Isolate least significant bit
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        # The image of the union is the union of the images
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
        
        # Check if singleton (synchronized)
        if cur & (cur - 1) == 0:
            return dist[cur]
        
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
            
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
            
    return 0

def _generate_ping_pong_structures():
    """
    Generate a set of candidate automata based on known high-threshold structures.
    Focuses on 'Pinwheel' and 'Ping-Pong' variants.
    """
    candidates = []
    
    # 1. Pinwheel structures
    # a is a cycle: 0->1->...->N-1->0
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # b is a 'shift' map that synchronizes.
    # Classic pinwheel: b maps i to i-1 (mod N) but with a twist.
    # Or: b maps a range of states to a single state, shifting the "gap".
    
    # Variant 1: b(i) = (i - 1) % N is a cycle, doesn't sync alone.
    # We need b to be non-invertible.
    # Let's try: b(0)=0, b(i)=i-1 for i>0. This is a 'tree' structure.
    # Threshold for this is usually N-1. Not high enough.
    
    # Better Pinwheel:
    # a: 0->1, 1->2, ..., N-1->0
    # b: 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2
    # This creates a "sliding window" effect.
    b_std = [0] * N
    b_std[0] = 0
    for i in range(1, N):
        b_std[i] = i - 1
    candidates.append((a_cycle[:], b_std[:]))
    
    # Variant 2: Shift the 'sink' of b
    # b(0)=1, b(1)=0, b(2)=1, b(3)=2, ...
    # Or more generally, b(i) = (i-1) % N for i != 0, b(0) = 0 is the above.
    # Let's try varying the 'break' point in the cycle for b.
    # b maps N-1 states to N-2 states in a chain.
    
    for k in range(N):
        b_var = [0] * N
        # Create a chain of length N-1 that ends at k, and map one state to k as well?
        # Actually, a common high-threshold structure is:
        # a: cycle
        # b: maps i to i-1 (mod N) EXCEPT one state which maps to 0.
        # Let's try: b(i) = (i - 1) % N for all i, except b(k) = 0.
        # If k=0, b(0)=0, b(1)=0, b(2)=1... this is Variant 1.
        # If k!=0, b(k)=0, b(k-1)=k-2... 
        # Let's generate these.
        b_k = [0] * N
        for i in range(N):
            if i == k:
                b_k[i] = 0
            else:
                b_k[i] = (i - 1) % N
        candidates.append((a_cycle[:], b_k[:]))
        
    # 2. Ping-Pong structures
    # Two permutations? No, must synchronize, so at least one must be non-invertible.
    # But often both are 'almost' permutations.
    
    # Let's try:
    # a: 0->1, 1->0, 2->3, 3->2, ... (pairs)
    # b: 0->0, 1->0, 2->1, 3->2, ...
    a_pairs = [0] * N
    for i in range(0, N, 2):
        a_pairs[i] = i + 1
        a_pairs[i+1] = i
    b_chain = [0] * N
    for i in range(1, N):
        b_chain[i] = i - 1
    candidates.append((a_pairs[:], b_chain[:]))
    
    # 3. Randomized Perturbations of the Best Standard
    # Take the standard pinwheel and randomly swap destinations.
    base_a = a_cycle[:]
    base_b = b_std[:]
    for _ in range(20):
        a_pert = base_a[:]
        b_pert = base_b[:]
        # Swap two destinations in a
        i, j = random.sample(range(N), 2)
        a_pert[i], a_pert[j] = a_pert[j], a_pert[i]
        # Swap two destinations in b
        i, j = random.sample(range(N), 2)
        b_pert[i], b_pert[j] = b_pert[j], b_pert[i]
        candidates.append((a_pert, b_pert))
        
    return candidates

def _local_search(a, b, start_time, deadline, max_iterations=1000):
    """
    Hill climbing local search.
    Tries to improve the threshold by single-edge changes.
    """
    cur_a = a[:]
    cur_b = b[:]
    cur_score = compute_threshold(cur_a, cur_b)
    
    if cur_score == 0:
        return None, None, 0
        
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Choose a random mutation
        # 50% chance mutate a, 50% chance mutate b
        if random.random() < 0.5:
            state = random.randint(0, N-1)
            old_dest = cur_a[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            cur_a[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            
            if new_score > cur_score:
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_a[state] = old_dest
                no_improve += 1
        else:
            state = random.randint(0, N-1)
            old_dest = cur_b[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
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
                cur_b[state] = old_dest
                no_improve += 1
                
    return best_a, best_b, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Use a fixed seed for reproducibility of the "search" part, 
    # but the initial structures are deterministic.
    random.seed(42)
    
    best_a = None
    best_b = None
    best_score = 0
    
    # Phase 1: Evaluate structural candidates
    candidates = _generate_ping_pong_structures()
    
    for a_cand, b_cand in candidates:
        if time.time() > deadline - 10.0:
            break
        score = compute_threshold(a_cand, b_cand)
        if score > best_score:
            best_score = score
            best_a = a_cand[:]
            best_b = b_cand[:]
            
    # If we hit the theoretical max, return early
    if best_score == (N-1)**2:
        return [best_a, best_b]

    # Phase 2: Local search from the best candidate
    if best_a is not None:
        refined_a, refined_b, refined_score = _local_search(
            best_a, best_b, start_time, deadline - 5.0
        )
        if refined_score > best_score:
            best_score = refined_score
            best_a = refined_a
            best_b = refined_b
            
    if best_score == (N-1)**2:
        return [best_a, best_b]

    # Phase 3: Try local search from other promising candidates
    # Sort candidates by initial score? We didn't store them all. 
    # Let's just do a few random restarts if time permits.
    
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try to improve the best found so far with more aggressive search?
        # Or try a few random perturbations of the best found.
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb the best known automaton
            a_pert = best_a[:]
            b_pert = best_b[:]
            # Make 2-3 random changes
            for _ in range(3):
                if random.random() < 0.5:
                    i = random.randint(0, N-1)
                    a_pert[i] = random.randint(0, N-1)
                else:
                    i = random.randint(0, N-1)
                    b_pert[i] = random.randint(0, N-1)
            
            score = compute_threshold(a_pert, b_pert)
            if score > best_score:
                best_score = score
                best_a = a_pert[:]
                best_b = b_pert[:]
                
            # If this perturbation is good, refine it
            if score >= best_score - 5:
                refined_a, refined_b, refined_score = _local_search(
                    a_pert, b_pert, start_time, deadline - 2.0, max_iterations=200
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_a = refined_a
                    best_b = refined_b
                    
            if best_score == (N-1)**2:
                return [best_a, best_b]

    # Fallback: ensure we return a valid synchronizing automaton
    if best_score == 0 or best_a is None:
        # Return the standard pinwheel which is guaranteed to sync
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        b[0] = 0
        for i in range(1, N):
            b[i] = i - 1
        return [a, b]
    
    return [best_a, best_b]

# Optional: Test locally
if __name__ == "__main__":
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"A: {a}")
    print(f"B: {b}")
