"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Pre-compute GCDs and prime factors for speed
    all_moduli = list(range(20, 201))
    num_moduli = len(all_moduli)
    
    # gcd_matrix[i][j] = gcd(m_i, m_j)
    gcd_matrix = [[0] * num_moduli for _ in range(num_moduli)]
    for i in range(num_moduli):
        m_i = all_moduli[i]
        for j in range(i, num_moduli):
            m_j = all_moduli[j]
            g = math.gcd(m_i, m_j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    # Pre-compute valid residues for each modulus against all others? 
    # No, because validity depends on the specific residues chosen by others.
    # However, we can pre-compute the "conflict graph" structure if residues were fixed, 
    # but here residues are variables.
    
    # We will use a Simulated Annealing approach.
    # State: A mapping from modulus_index -> residue (or -1 if not selected)
    # Since we want to maximize sum(1/m), and all 1/m are positive, 
    # it is generally beneficial to select as many moduli as possible, 
    # especially the small ones. 
    # However, selecting a modulus might block multiple others.
    # A common heuristic is to try to select ALL moduli if possible, 
    # or a large subset.
    
    # Let's define the search space as assignments for a fixed set of candidate moduli.
    # To ensure we find a good solution, we should include all moduli in the candidate set 
    # and allow the algorithm to decide which ones to keep (by setting residue to -1) 
    # or which residue to use.
    
    # Actually, a simpler and more effective SA for this specific problem:
    # 1. Start with a greedy solution.
    # 2. Define a move: Pick a random modulus in the solution, change its residue to a random valid one (if any), 
    #    or pick a modulus NOT in the solution and try to add it, or remove a modulus.
    # 3. Score is sum(1/m). 
    # 4. Constraint: Pairwise disjointness.
    
    # Efficient Conflict Checking:
    # For a given state (list of [m, r]), checking if a new [m_new, r_new] is valid requires 
    # checking against all existing. O(N).
    # Changing an existing residue requires checking against all others. O(N).
    # Removing an element is O(1) for the list, but score update is O(1).
    
    # To make SA fast, we need to update the score and validity quickly.
    # Let's represent the state as a list of tuples (m_idx, r).
    # m_idx is index in all_moduli.
    
    def get_score(state):
        # state: list of (m_idx, r)
        s = 0.0
        for m_idx, r in state:
            s += 1.0 / all_moduli[m_idx]
        return s

    def is_valid_state(state):
        # Check pairwise disjointness
        n = len(state)
        for i in range(n):
            m1_idx, r1 = state[i]
            m1 = all_moduli[m1_idx]
            for j in range(i + 1, n):
                m2_idx, r2 = state[j]
                m2 = all_moduli[m2_idx]
                g = gcd_matrix[m1_idx][m2_idx]
                if (r1 - r2) % g == 0:
                    return False
        return True

    def generate_greedy_state():
        # Start with empty, add moduli in order of increasing m (high value)
        # If a conflict occurs, try random residues. If stuck, skip.
        state = []
        for m_idx in range(num_moduli):
            m = all_moduli[m_idx]
            # Try to find a valid residue
            valid_r = -1
            # Shuffled range for randomness in tie-breaking
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                conflict = False
                for m2_idx, r2 in state:
                    g = gcd_matrix[m_idx][m2_idx]
                    if (r - r2) % g == 0:
                        conflict = True
                        break
                if not conflict:
                    valid_r = r
                    break
            if valid_r != -1:
                state.append((m_idx, valid_r))
        return state

    # Simulated Annealing
    best_state = generate_greedy_state()
    best_score = get_score(best_state)
    
    # Ensure the initial state is valid (it should be by construction)
    if not is_valid_state(best_state):
        # Fallback if greedy fails (unlikely)
        best_state = []
        best_score = 0.0

    current_state = best_state[:]
    current_score = best_score
    
    # SA Parameters
    T = 1.0  # Initial temperature
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    
    # We need to estimate the number of steps we can do.
    # Each step involves:
    # 1. Picking a move.
    # 2. Updating state.
    # 3. Calculating delta score.
    # 4. Checking validity (or updating conflict counts).
    
    # To speed up validity checks, we can maintain a conflict count matrix or similar?
    # Given N=181, O(N^2) check is ~32k operations. 
    # If we do 10k steps, that's 320M ops. Might be too slow in Python.
    # We need a faster validity check for moves.
    
    # Alternative: 
    # When changing residue of item i to new_r:
    # It is valid iff for all j != i, (new_r - r_j) % gcd(m_i, m_j) != 0.
    # This is O(N) per move.
    # Total steps * N. If we can do 50k steps, 50k * 181 = 9M ops. This is feasible in Python (<1s).
    
    # Let's implement the move logic carefully.
    
    def try_move(state, score):
        """
        Attempts a random move. Returns (new_state, new_score, delta_score, valid)
        """
        n = len(state)
        if n == 0:
            # Add a random modulus
            m_idx = random.randint(0, num_moduli - 1)
            m = all_moduli[m_idx]
            r = random.randint(0, m - 1)
            # Check validity
            valid = True
            for m2_idx, r2 in state:
                g = gcd_matrix[m_idx][m2_idx]
                if (r - r2) % g == 0:
                    valid = False
                    break
            if valid:
                new_state = state + [(m_idx, r)]
                new_score = score + 1.0 / m
                return new_state, new_score, 1.0 / m, True
            else:
                return state, score, 0.0, False

        # Choose move type:
        # 1. Change residue of an existing element (50%)
        # 2. Add a new element (25%)
        # 3. Remove an element (25%)
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Change residue
            i = random.randint(0, n - 1)
            m_idx, old_r = state[i]
            m = all_moduli[m_idx]
            
            # Try a few random new residues
            for _ in range(10): # Limit attempts per move to avoid stalling
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    m2_idx, r2 = state[j]
                    g = gcd_matrix[m_idx][m2_idx]
                    if (new_r - r2) % g == 0:
                        valid = False
                        break
                if valid:
                    new_state = state[:]
                    new_state[i] = (m_idx, new_r)
                    # Score doesn't change
                    return new_state, score, 0.0, True
            return state, score, 0.0, False

        elif move_type < 0.75:
            # Add element
            # Pick a modulus not in state
            in_state = set(m_idx for m_idx, r in state)
            candidates = [m_idx for m_idx in range(num_moduli) if m_idx not in in_state]
            if not candidates:
                return state, score, 0.0, False
            
            m_idx = random.choice(candidates)
            m = all_moduli[m_idx]
            
            # Try to find a valid residue
            # Instead of random, we can try to find one that doesn't conflict.
            # But for SA, we just need a valid move.
            # We'll try a few random residues.
            for _ in range(20):
                new_r = random.randint(0, m - 1)
                valid = True
                for m2_idx, r2 in state:
                    g = gcd_matrix[m_idx][m2_idx]
                    if (new_r - r2) % g == 0:
                        valid = False
                        break
                if valid:
                    new_state = state + [(m_idx, new_r)]
                    new_score = score + 1.0 / m
                    return new_state, new_score, 1.0 / m, True
            return state, score, 0.0, False
            
        else:
            # Remove element
            i = random.randint(0, n - 1)
            m_idx, r = state[i]
            new_state = state[:i] + state[i+1:]
            new_score = score - 1.0 / all_moduli[m_idx]
            return new_state, new_score, -1.0 / all_moduli[m_idx], True

    # Main SA Loop
    steps = 0
    while time.time() < deadline:
        new_state, new_score, delta, valid = try_move(current_state, current_score)
        
        if not valid:
            T *= alpha
            continue
            
        if delta >= 0:
            # Accept
            current_state = new_state
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_state = current_state[:]
        else:
            # Reject with probability exp(-delta / T)
            # delta is negative, so -delta/T is positive
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_state = new_state
                current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: restart with higher T if stuck? 
            # Or just keep going.
            
        steps += 1
        if steps % 10000 == 0:
            # Check time frequently
            pass

    # Final validation
    if not is_valid_state(best_state):
        # Fallback to greedy
        best_state = generate_greedy_state()
        if not is_valid_state(best_state):
             best_state = []

    # Convert to required format [[m, r], ...]
    result = []
    for m_idx, r in best_state:
        result.append([all_moduli[m_idx], r])
        
    return result

# EVOLVE-BLOCK-END
