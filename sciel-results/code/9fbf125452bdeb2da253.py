import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def evaluate(coeffs):
        if not is_valid(coeffs):
            return float('inf')
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    best_score = float('inf')
    best_poly = [1] * 12

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if not is_valid(coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Try ILP first
    try:
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix
    except ImportError:
        linprog = None

    if linprog is not None and time.time() < deadline - 30:
        # ILP formulation
        # Variables:
        #   x[0..11]: coefficient of term i (continuous, but we'll use bounds and hope for integer)
        #   Actually, let's use a different approach.
        #   We want to minimize sum of indicators for nonzero squared terms.
        #   This is a mixed-integer quadratic program, which is hard.
        #   Let's instead use a simpler heuristic: for each support, solve for signs/mags that minimize.
        
        # Alternative: exhaustive search over supports with small degree, using ILP for signs.
        # For a fixed support S of 12 positions, the square has terms at positions i+j.
        # The coefficient at k is sum_{i+j=k, i,j in S} c_i * c_j.
        # We want to choose c_i in [-64,64] nonzero to minimize count of k where this sum != 0.
        
        # This is still hard. Let's try a different structural approach:
        # Use the fact that if P(x) = Q(x)^2 * R(x) where Q is sparse, then P is a square.
        # But P must have exactly 12 terms.
        
        # Let's try: P(x) = A(x) * B(x) where A and B are sparse and A^2 = B^2, so P is a square.
        # Actually, simpler: let's just do a more efficient local search with better initial states.
        
        pass

    # More efficient approach: precompute all possible supports and use vectorized evaluation
    # Key insight: for a given support, the square's nonzero count depends on the overlap structure.
    # Two terms at positions i and j contribute to square positions i+i, i+j, j+i, j+j.
    # If i+j = k+l for distinct pairs, cancellation can occur.
    
    # Let's try a structured approach: search for supports where many pair sums collide.
    # A "collision" at position k means multiple pairs (i,j) with i+j=k.
    # If we can make the coefficients cancel at collision points, we reduce nonzero count.
    
    # Strategy: find supports with maximum number of pair-sum collisions, then assign signs to cancel.
    
    def count_pair_sum_collisions(support):
        """Count how many pair sums have multiplicity > 1."""
        from collections import Counter
        pair_sums = []
        for i in range(len(support)):
            for j in range(i, len(support)):
                pair_sums.append(support[i] + support[j])
        c = Counter(pair_sums)
        collisions = sum(1 for v in c.values() if v > 1)
        return collisions, c

    # Search for supports with high collision counts
    # But we need to also be able to cancel, which requires the pairs to have matching structure.
    
    # Let's try a different angle: use the polynomial identity
    # (a + b x^d)^2 = a^2 + 2ab x^d + b^2 x^{2d}  -- 3 terms
    # (a + b x^d + c x^{2d})^2 -- 5 terms if no cancellation
    # With cancellation: (a + b x^d - c x^{2d})^2 where 2ab = 0? No.
    
    # Actually, let's try: P(x) = (sum of 6 terms)^2 is not what we want.
    # We want P to have 12 terms and P^2 to be sparse.
    
    # Let's just do a very fast local search with better moves.
    
    def make_state(pos_sign_mag_list):
        sorted_list = sorted(pos_sign_mag_list, key=lambda x: x[0])
        return tuple(sorted_list)

    def state_score_fast(state):
        positions = [s[0] for s in state]
        signs = [s[1] for s in state]
        mags = [s[2] for s in state]
        max_deg = max(positions)
        coeffs = np.zeros(max_deg + 1, dtype=np.int64)
        for p, s, m in state:
            coeffs[p] = s * m
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def state_to_coeffs(state):
        positions = [s[0] for s in state]
        max_deg = max(positions)
        coeffs = [0] * (max_deg + 1)
        for p, s, m in state:
            coeffs[p] = s * m
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    # Generate diverse initial states
    initial_states = []
    
    # Consecutive
    initial_states.append(make_state([(i, 1, 1) for i in range(12)]))
    
    # Alternating signs consecutive
    initial_states.append(make_state([(i, 1 if i % 2 == 0 else -1, 1) for i in range(12)]))
    
    # Geometric spacing
    for ratio in [2, 3]:
        pos = [0]
        for i in range(1, 12):
            pos.append(min(pos[-1] * ratio, 40))
        pos = sorted(set(pos))
        while len(pos) < 12:
            pos.append(random.randint(0, 40))
        pos = sorted(set(pos))[:12]
        if len(pos) == 12:
            initial_states.append(make_state([(p, 1, 1) for p in pos]))
    
    # Random with bias toward collisions
    for _ in range(50):
        pos = sorted(random.sample(range(41), 12))
        signs = [random.choice([-1, 1]) for _ in range(12)]
        mags = [random.choice([1, 1, 1, 2, 3, 4]) for _ in range(12)]
        initial_states.append(make_state([(pos[i], signs[i], mags[i]) for i in range(12)]))
    
    # Evaluate
    best_state = None
    best_state_score = float('inf')
    for state in initial_states:
        positions = [s[0] for s in state]
        if len(set(positions)) != 12:
            continue
        if any(p < 0 or p > 40 for p in positions):
            continue
        cnt = state_score_fast(state)
        if cnt < best_state_score:
            best_state_score = cnt
            best_state = state
        coeffs = state_to_coeffs(state)
        if is_valid(coeffs):
            try_poly(coeffs)
    
    if best_state is None:
        best_state = initial_states[0]
        best_state_score = state_score_fast(best_state)
    
    # Fast local search
    current_state = best_state
    current_score = best_state_score
    best_ls_state = current_state
    best_ls_score = current_score
    
    steps = 0
    max_steps = 200000
    while steps < max_steps and time.time() < deadline - 5:
        steps += 1
        state_list = list(current_state)
        move_type = random.randint(1, 6)
        
        if move_type == 1:
            i = random.randint(0, 11)
            new_pos = random.randint(0, 40)
            if new_pos not in [s[0] for s in state_list if s[0] != state_list[i][0]]:
                state_list[i] = (new_pos, state_list[i][1], state_list[i][2])
        elif move_type == 2:
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pj, si, mi)
            state_list[j] = (pi, sj, mj)
        elif move_type == 3:
            i = random.randint(0, 11)
            p, s, m = state_list[i]
            state_list[i] = (p, -s, m)
        elif move_type == 4:
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pi, -si, mi)
            state_list[j] = (pj, -sj, mj)
        elif move_type == 5:
            i = random.randint(0, 11)
            p, s, m = state_list[i]
            new_m = random.randint(1, 8)
            state_list[i] = (p, s, new_m)
        elif move_type == 6:
            i = random.randint(0, 11)
            new_pos = random.randint(0, 40)
            p, s, m = state_list[i]
            if new_pos not in [x[0] for x in state_list if x[0] != p]:
                state_list[i] = (new_pos, -s, m)
        
        new_state = make_state(state_list)
        positions = [s[0] for s in new_state]
        if len(set(positions)) != 12:
            continue
        if any(p < 0 or p > 40 for p in positions):
            continue
        
        new_score = state_score_fast(new_state)
        if new_score <= current_score:
            current_state = new_state
            current_score = new_score
            if new_score < best_ls_score:
                best_ls_score = new_score
                best_ls_state = new_state
                coeffs = state_to_coeffs(new_state)
                if is_valid(coeffs):
                    try_poly(coeffs)
        elif random.random() < 0.1:  # occasional random restart of current
            current_state = new_state
            current_score = new_score
    
    # Final greedy improvement
    if time.time() < deadline - 3:
        current_state = best_ls_state
        current_score = best_ls_score
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(12):
                if time.time() > deadline - 3:
                    break
                p, s, m = current_state[i]
                # Try sign flip
                state_list = list(current_state)
                state_list[i] = (p, -s, m)
                new_state = make_state(state_list)
                new_score = state_score_fast(new_state)
                if new_score < current_score:
                    current_state = new_state
                    current_score = new_score
                    improved = True
                    coeffs = state_to_coeffs(new_state)
                    if is_valid(coeffs):
                        try_poly(coeffs)
                # Try magnitude change
                for new_m in range(1, 9):
                    if new_m == m:
                        continue
                    state_list = list(current_state)
                    state_list[i] = (p, s, new_m)
                    new_state = make_state(state_list)
                    new_score = state_score_fast(new_state)
                    if new_score < current_score:
                        current_state = new_state
                        current_score = new_score
                        improved = True
                        coeffs = state_to_coeffs(new_state)
                        if is_valid(coeffs):
                            try_poly(coeffs)
                        break
                if time.time() > deadline - 3:
                    break
                # Try position move
                for new_pos in range(41):
                    if new_pos == p:
                        continue
                    if new_pos in [s2[0] for s2 in current_state if s2[0] != p]:
                        continue
                    state_list = list(current_state)
                    state_list[i] = (new_pos, s, m)
                    new_state = make_state(state_list)
                    new_score = state_score_fast(new_state)
                    if new_score < current_score:
                        current_state = new_state
                        current_score = new_score
                        improved = True
                        coeffs = state_to_coeffs(new_state)
                        if is_valid(coeffs):
                            try_poly(coeffs)
                        break
    
    if not is_valid(best_poly):
        best_poly = [1] * 12
    
    return best_poly
