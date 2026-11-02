import time
import sys
import math
import random
from collections import defaultdict

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate all permutations
    import itertools
    perms = list(itertools.permutations(range(1, n + 1)))
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Build the rotation graph
    # next_perm[perm] = left rotation of perm
    # The graph is a set of cycles.
    visited = [False] * len(perms)
    cycles = []
    
    for i in range(len(perms)):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                # Next permutation is left rotation
                p = perms[curr]
                rot = p[1:] + p[:1]
                curr = perm_to_idx[rot]
            cycles.append(cycle)
            
    # For each cycle, generate the words for each starting offset
    # Cycle length is n (for prime n)
    cycle_words = []
    for cycle in cycles:
        words_for_cycle = []
        for start_offset in range(n):
            # Construct word starting from cycle[start_offset]
            # Traverse n steps
            word = []
            for step in range(n):
                idx = cycle[(start_offset + step) % n]
                p = perms[idx]
                if step == 0:
                    word.extend(p)
                else:
                    word.append(p[-1])
            words_for_cycle.append(tuple(word))
        cycle_words.append(words_for_cycle)
        
    # Precompute overlaps between all pairs of blocks
    # Block is (cycle_idx, offset)
    # Total blocks = len(cycles) * n
    num_cycles = len(cycles)
    num_blocks = num_cycles * n
    
    # Overlap matrix: overlap[block1][block2] = max overlap length
    # To save memory/time, we compute on the fly or precompute a smaller structure
    # Since num_blocks ~ 720 * 7 = 5040, a 5040x5040 matrix is 25M entries.
    # In Python, this might be slow/memory heavy.
    # Let's compute overlap on the fly in the SA evaluation.
    
    def get_overlap(word1, word2):
        # word1, word2 are tuples of ints
        # Max overlap is n-1
        max_ov = min(n - 1, len(word1), len(word2))
        for ov in range(max_ov, -1, -1):
            if ov == 0:
                return 0
            if word1[-ov:] == word2[:ov]:
                return ov
        return 0

    # Simulated Annealing
    # State: list of block indices (0 to num_blocks-1)
    # block_idx = cycle_idx * n + offset
    
    # Initialize state: random permutation of blocks
    state = list(range(num_blocks))
    random.shuffle(state)
    
    def evaluate(state):
        total_len = 0
        prev_word = None
        for b_idx in state:
            c_idx = b_idx // n
            offset = b_idx % n
            word = cycle_words[c_idx][offset]
            if prev_word is None:
                total_len += len(word)
            else:
                ov = get_overlap(prev_word, word)
                total_len += len(word) - ov
            prev_word = word
        return total_len

    # SA Parameters
    start_time = time.time()
    time_limit = 110.0
    T = 100.0
    cooling = 0.999
    best_state = state[:]
    best_score = evaluate(best_state)
    current_score = best_score
    
    while time.time() - start_time < time_limit:
        # Move: Swap two random blocks
        i = random.randint(0, num_blocks - 1)
        j = random.randint(0, num_blocks - 1)
        if i == j:
            continue
            
        state[i], state[j] = state[j], state[i]
        
        new_score = evaluate(state)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_state = state[:]
                
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    # Reconstruct word from best_state
    word = []
    prev_word = None
    for b_idx in best_state:
        c_idx = b_idx // n
        offset = b_idx % n
        w = cycle_words[c_idx][offset]
        if prev_word is None:
            word.extend(w)
        else:
            ov = get_overlap(prev_word, w)
            word.extend(w[ov:])
        prev_word = w
        
    return "".join(map(str, word))
