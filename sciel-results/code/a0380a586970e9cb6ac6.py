import time
import random
import numpy as np
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _compute_threshold_fast(trans0, trans1):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Uses bitmasks for fast subset transitions.
    Returns -1 if not synchronizing.
    """
    # Precompute transition masks for each state and letter
    t0 = [1 << trans0[i] for i in range(N)]
    t1 = [1 << trans1[i] for i in range(N)]
    
    # Helper to compute next subset
    def get_next(subset, letter):
        if letter == 0:
            res = 0
            s = subset
            while s:
                lsb = s & -s
                idx = (lsb.bit_length() - 1)
                res |= t0[idx]
                s ^= lsb
            return res
        else:
            res = 0
            s = subset
            while s:
                lsb = s & -s
                idx = (lsb.bit_length() - 1)
                res |= t1[idx]
                s ^= lsb
            return res

    # Precompute all transitions to make BFS O(1) per step
    next_subsets = [[0, 0] for _ in range(1 << N)]
    for s in range(1 << N):
        next_subsets[s][0] = get_next(s, 0)
        next_subsets[s][1] = get_next(s, 1)
        
    # BFS
    visited = [-1] * (1 << N)
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Check if singleton
        if current & (current - 1) == 0 and current != 0:
            return dist
            
        for letter in range(2):
            nxt = next_subsets[current][letter]
            if visited[nxt] == -1:
                visited[nxt] = dist + 1
                queue.append(nxt)
                
    return -1

def _generate_cerny_seed():
    """
    Generates a standard Cerny automaton which has threshold N-1.
    a: cycle (0->1->...->N-1->0)
    b: 0->0, i->i-1 for i>0
    """
    trans0 = [(i + 1) % N for i in range(N)]
    trans1 = [0] * N
    for i in range(1, N):
        trans1[i] = i - 1
    return trans0, trans1

def _generate_near_permutation_seed(rng):
    """
    Generates a seed where one letter is a random permutation (cycle-like)
    and the other is a 'defective' permutation (mostly shift, some random breaks).
    This structure is known to produce longer reset words.
    """
    # Letter 0: Random permutation
    t0 = list(range(N))
    rng.shuffle(t0)
    
    # Letter 1: Start with a shift, then introduce 1-3 random defects
    shift = rng.randint(1, N - 1)
    t1 = [(i + shift) % N for i in range(N)]
    
    # Introduce defects to break the cycle and allow synchronization
    num_defects = rng.randint(1, 3)
    for _ in range(num_defects):
        idx = rng.randint(0, N - 1)
        # Change destination to something else, preferably closer to 'sink' behavior
        # or just random to break symmetry
        t1[idx] = rng.randint(0, N - 1)
        
    return t0, t1

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin for safety
    
    rng = random.Random(42)
    
    # Multi-start Simulated Annealing
    # We will run several independent SA chains.
    
    best_0_global = None
    best_1_global = None
    best_score_global = 0
    
    # Number of independent restarts/chains
    num_chains = 10
    
    for chain_idx in range(num_chains):
        # Check time
        if time.time() > deadline:
            break
            
        # Initialize chain
        if chain_idx == 0:
            current_0, current_1 = _generate_cerny_seed()
        else:
            # Mix of Cerny seeds and near-permutation seeds
            if rng.random() < 0.5:
                current_0, current_1 = _generate_cerny_seed()
            else:
                current_0, current_1 = _generate_near_permutation_seed(rng)
                
        current_score = _compute_threshold_fast(current_0, current_1)
        if current_score < 0:
            current_score = 0 # Treat non-sync as 0 for SA purposes, but we must return sync.
            # Actually, if it doesn't sync, we should probably restart this chain immediately
            # or just accept a very low score and hope SA fixes it.
            # Better: restart chain if not sync.
            if current_score < 0:
                continue

        best_0_chain = current_0[:]
        best_1_chain = current_1[:]
        best_score_chain = current_score
        
        # SA Parameters
        # Temperature schedule: T = T0 * cooling^iteration
        T = 50.0
        T_min = 0.1
        cooling = 0.9995
        max_iters = 50000 # Hard cap per chain, but time is the real limit
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Mutation: Change one transition of one letter
            mut_0 = current_0[:]
            mut_1 = current_1[:]
            
            letter = rng.randint(0, 1)
            state = rng.randint(0, N - 1)
            
            if letter == 0:
                old_val = mut_0[state]
                new_val = rng.randint(0, N - 1)
                if new_val == old_val:
                    continue
                mut_0[state] = new_val
                new_score = _compute_threshold_fast(mut_0, mut_1)
            else:
                old_val = mut_1[state]
                new_val = rng.randint(0, N - 1)
                if new_val == old_val:
                    continue
                mut_1[state] = new_val
                new_score = _compute_threshold_fast(mut_0, mut_1)
            
            # Handle non-synchronizing automata
            # If new is not sync, score is -1. We treat this as a huge penalty.
            # For SA, we need a consistent energy scale.
            # Let's define energy = -score. Lower is better.
            # If score is -1, energy is +100 (penalty).
            # If score is 11, energy is -11.
            
            old_energy = -current_score if current_score >= 0 else 100
            new_energy = -new_score if new_score >= 0 else 100
            
            delta_energy = new_energy - old_energy
            
            accept = False
            if delta_energy < 0:
                accept = True
            else:
                # Metropolis criterion
                if T > T_min:
                    # P = exp(-delta / T)
                    # Avoid overflow/underflow
                    if delta_energy < 100: # Only accept if not insanely bad
                        p = np.exp(-delta_energy / T)
                        if rng.random() < p:
                            accept = True
            
            if accept:
                current_0, current_1 = mut_0, mut_1
                current_score = new_score
                
                if current_score > best_score_chain:
                    best_score_chain = current_score
                    best_0_chain = current_0[:]
                    best_1_chain = current_1[:]
                    
                    if best_score_chain > best_score_global:
                        best_score_global = best_score_chain
                        best_0_global = best_0_chain[:]
                        best_1_global = best_1_chain[:]
                        
            T *= cooling

    # Fallback if no valid sync automata found (should not happen with Cerny seed)
    if best_0_global is None:
        b0, b1 = _generate_cerny_seed()
        return [b0, b1]
        
    return [best_0_global, best_1_global]

# EVOLVE-BLOCK-END
