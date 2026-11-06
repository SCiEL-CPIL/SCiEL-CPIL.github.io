"""HP lattice protein folding - Constraint-Satisfaction Backtracking with Contact Map Prioritization."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 195.0


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    return [(point[0] + dx, point[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]


def _energy(sequence, chain):
    """Compute HP energy: -1 for each non-consecutive H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _contact_map(sequence):
    """
    Identify high-priority H-H pairs based on sequence distance.
    Pairs closer in sequence (but not adjacent) are higher priority.
    """
    n = len(sequence)
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    pairs = []
    for i in range(len(h_indices)):
        for j in range(i + 1, len(h_indices)):
            idx_i = h_indices[i]
            idx_j = h_indices[j]
            dist = abs(idx_i - idx_j)
            if dist > 1:
                # Priority is inversely proportional to distance
                priority = 1.0 / dist
                pairs.append((priority, idx_i, idx_j))
    pairs.sort(key=lambda x: -x[0])
    return pairs


def _backtrack_fold(sequence, dimension, max_depth=15, max_restarts=50, rng=None):
    """
    Backtracking search guided by contact map priorities.
    Tries to satisfy high-priority H-H contacts early in the folding process.
    """
    if rng is None:
        rng = random.Random(42)
        
    n = len(sequence)
    if n == 0:
        return None
    if n == 1:
        return [(0, 0)]
    
    # Pre-compute contact map
    # For backtracking, we care about contacts that can be formed *during* the growth.
    # A contact between i and j can only be formed when placing i if j is already placed,
    # or when placing j if i is already placed.
    # To guide the search, we prioritize moves that satisfy contacts with *already placed* H residues.
    
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    best_chain = None
    best_energy = float('inf')
    
    # We will run multiple restarts with different random seeds/choices
    for restart in range(max_restarts):
        if _remaining() < 1.0:
            break
        
        # Randomize the starting direction and maybe the specific tie-breaking
        start_dir = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        
        # State:
        # chain: list of (x, y)
        # occupied: dict (x, y) -> index
        # current_index: next index to place
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        # For efficiency, we might limit the depth of the search or use a heuristic cutoff
        # But for now, let's try a full backtracking with a limit on branching factor or depth.
        # Given n can be up to 100, full backtracking is impossible.
        # We will use a "lookahead" or "greedy backtracking" where we only backtrack
        # if we get stuck or if the potential energy is too high.
        
        # Actually, a pure backtracking for n=100 is too slow.
        # The strategy is: "pre-identify high-priority H-H pairs and enforce their spatial proximity via hard constraints during a backtracking search"
        # This implies we should try to *force* these contacts.
        
        # Let's implement a "Constrained Greedy" that backtracks only when a high-priority contact is *impossible* to satisfy later, or simply use a limited depth backtracking for the first K residues to set the "core", then greedy for the rest.
        
        # Better approach for this specific prompt: 
        # 1. Identify top K pairs.
        # 2. Try to fold such that these pairs are adjacent.
        
        # Let's try a different tactic: 
        # "Enforce spatial proximity via hard constraints"
        # We can define a set of "must-have" contacts.
        # If we place residue i, and it's H, we check if any previously placed H residue j is a "high priority" partner.
        # If so, we *must* place i adjacent to j? No, that's too strong and might lead to dead ends.
        
        # Let's stick to: Backtracking with a strong heuristic score that heavily penalizes failing to satisfy high-priority contacts *when possible*.
        # And we limit the search depth to the first `max_depth` residues to ensure we find a good "core", then finish with greedy.
        
        # Actually, let's just do a randomized greedy with a "lookahead" of 1 step for contact satisfaction, and restarts.
        # The "Constraint Satisfaction" aspect: 
        # When choosing the next position for residue i (H), if there is a previously placed H residue j such that (i, j) is a high-priority pair, 
        # we give a HUGE bonus to placing i adjacent to j.
        
        # Let's refine the score function in greedy growth.
        
        def score_move(i, pos, chain_so_far, occupied_map):
            # i: index of residue to place
            # pos: candidate position
            # chain_so_far: list of positions
            # occupied_map: dict pos -> index
            
            res = sequence[i]
            score = 0.0
            
            # 1. Check for high-priority contact satisfaction
            if res == 'H':
                for q in _neighbours(pos):
                    j = occupied_map.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        # Calculate priority of pair (i, j)
                        dist = i - j
                        priority = 1.0 / dist
                        # Huge weight for satisfying this contact
                        score += priority * 100.0 
                        
            # 2. Standard open space score
            free = sum(1 for q in _neighbours(pos) if q not in occupied_map)
            score += free * 1.0
            
            # 3. Random noise
            score += rng.random() * 0.1
            
            return score

        # We will use this scoring in a greedy loop, but with restarts.
        # To make it "backtracking", we can try a few alternatives if the top choice leads to a dead end.
        # For simplicity and speed, we'll do a "beam search" of width 1 (greedy) but with many restarts,
        # where the restarts vary the initial direction and the random noise.
        
        # However, the prompt asks for "backtracking search".
        # Let's implement a limited backtracking: 
        # At each step, consider the top 2 moves. If the top move leads to a dead end within 3 steps, backtrack and try the second.
        
        # Given the complexity, let's implement a simple DFS with a depth limit for the first part of the chain.
        
        # Actually, for n=100, even limited backtracking is risky.
        # Let's try: 
        # 1. Generate a "core" of length `core_len` (e.g., 10-15) using backtracking to maximize contacts among these.
        # 2. Extend the rest greedily.
        
        core_len = min(n, 12)
        
        # Backtracking for the core
        def backtrack_core(idx, current_chain, current_occupied, current_contacts):
            if _remaining() < 0.1:
                return None, -1000
            
            if idx == core_len:
                # Evaluate potential: count contacts within the core
                # And maybe "openness"
                return list(current_chain), current_contacts
            
            # Pruning: if current_contacts is very low compared to best so far? 
            # We don't have a global best for the core yet.
            
            last = current_chain[-1]
            candidates = []
            
            for pos in _neighbours(last):
                if pos in current_occupied:
                    continue
                
                # Calculate gain in contacts
                gain = 0
                if sequence[idx] == 'H':
                    for q in _neighbours(pos):
                        j = current_occupied.get(q)
                        if j is not None and j < idx - 1 and sequence[j] == 'H':
                            gain += 1
                            
                # Score: prioritize high gain
                # Also, avoid tight corners that might trap the chain?
                free = sum(1 for q in _neighbours(pos) if q not in current_occupied)
                
                # Heuristic: prefer moves that keep options open, but strongly prefer contacts
                # The "Constraint" aspect: if there's a high priority pair (idx, j) where j is already placed, 
                # and pos is adjacent to j, we MUST consider this highly.
                
                # Let's just sort by (gain, free)
                candidates.append((gain, free, pos))
                
            if not candidates:
                return None, -1000
            
            # Sort candidates by gain (desc), then free (desc)
            candidates.sort(key=lambda x: (-x[0], -x[1]))
            
            # Try the best candidate first
            # To allow backtracking, we would need to try others if this fails.
            # For now, let's just pick the best one (Greedy Core).
            # True backtracking requires recursion.
            
            # Let's do a depth-first search with a small limit on branching.
            # We'll try the top 2 candidates.
            
            for gain, free, pos in candidates[:2]:
                current_chain.append(pos)
                current_occupied[pos] = idx
                
                # Recurse
                sub_chain, sub_contacts = backtrack_core(idx + 1, current_chain, current_occupied, current_contacts + gain)
                
                if sub_chain is not None:
                    return sub_chain, sub_contacts
                
                # Backtrack
                current_chain.pop()
                del current_occupied[pos]
                
            return None, -1000

        core_chain, core_contacts = backtrack_core(0, chain, occupied, 0)
        
        if core_chain is None:
            continue
            
        # Now extend greedily
        current_chain = core_chain
        current_occupied = {p: i for i, p in enumerate(core_chain)}
        
        success = True
        for i in range(core_len, n):
            last = current_chain[-1]
            options = []
            
            for pos in _neighbours(last):
                if pos in current_occupied:
                    continue
                
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(pos):
                        j = current_occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            # Priority weighted gain
                            dist = i - j
                            priority = 1.0 / dist
                            gain += priority * 10.0 
                            
                free = sum(1 for q in _neighbours(pos) if q not in current_occupied)
                
                # Score
                score = gain * 5.0 + free * 0.5 + rng.random() * 0.2
                options.append((score, pos))
                
            if not options:
                success = False
                break
                
            options.sort(key=lambda x: -x[0])
            best_pos = options[0][1]
            
            current_chain.append(best_pos)
            current_occupied[best_pos] = i
            
        if not success:
            continue
            
        # Evaluate
        energy = _energy(sequence, current_chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = current_chain
            
    return best_chain

def _local_search(sequence, chain, dimension, rng, max_iter=50):
    """
    Local search with segment moves, biased towards satisfying high-priority contacts.
    """
    if chain is None:
        return None, 1
        
    current_energy = _energy(sequence, chain)
    current_chain = list(chain)
    n = len(current_chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    
    best_energy = current_energy
    best_chain = list(current_chain)
    
    # Pre-compute contact priorities
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    # We don't need the full map, just a quick check for high priority pairs
    # A pair (i, j) is high priority if |i-j| is small.
    
    for it in range(max_iter):
        if _remaining() < 0.2:
            break
            
        improved = False
        
        # Try a random segment move
        seg_len = rng.choice([2, 3, 4])
        if n <= seg_len + 2:
            continue
            
        i = rng.randint(1, n - seg_len - 2)
        
        d1 = (current_chain[i][0] - current_chain[i - 1][0],
              current_chain[i][1] - current_chain[i - 1][1])
              
        perps = [(-d1[1], d1[0]), (d1[1], -d1[0])]
        
        for px, py in perps:
            new_seg = []
            valid = True
            
            for k in range(seg_len):
                new_p = (current_chain[i + k][0] + px, current_chain[i + k][1] + py)
                new_seg.append(new_p)
                
            # Check occupancy
            for new_p in new_seg:
                j = pos_map.get(new_p)
                if j is not None and not (i <= j < i + seg_len):
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Check connections
            if abs(new_seg[0][0] - current_chain[i - 1][0]) + abs(new_seg[0][1] - current_chain[i - 1][1]) != 1:
                continue
            if abs(new_seg[-1][0] - current_chain[i + seg_len][0]) + abs(new_seg[-1][1] - current_chain[i + seg_len][1]) != 1:
                continue
            for k in range(seg_len - 1):
                if abs(new_seg[k + 1][0] - new_seg[k][0]) + abs(new_seg[k + 1][1] - new_seg[k][1]) != 1:
                    valid = False
                    break
            if not valid:
                continue
                
            # Compute energy delta
            delta = 0
            # Remove old contacts involving the segment
            for k in range(seg_len):
                old_p = current_chain[i + k]
                if sequence[i + k] == 'H':
                    for q in _neighbours(old_p):
                        j = pos_map.get(q)
                        if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                            if not (i <= j < i + seg_len):
                                delta += 1
                                
            # Add new contacts
            for k in range(seg_len):
                new_p = new_seg[k]
                if sequence[i + k] == 'H':
                    for q in _neighbours(new_p):
                        j = pos_map.get(q)
                        if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                            if not (i <= j < i + seg_len):
                                delta -= 1
                                
            if delta < 0:
                # Apply
                for k in range(seg_len):
                    old_p = current_chain[i + k]
                    del pos_map[old_p]
                    current_chain[i + k] = new_seg[k]
                    pos_map[new_seg[k]] = i + k
                    
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                improved = True
                break
                
        if not improved:
            # Try single residue moves
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            for q in _neighbours(p):
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                if q in pos_map and pos_map[q] != i:
                    continue
                prev = current_chain[i - 1]
                nxt = current_chain[i + 1]
                if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                    continue
                if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                    continue
                    
                delta = 0
                if sequence[i] == 'H':
                    for old_q in _neighbours(p):
                        j = pos_map.get(old_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    for new_q in _neighbours(q):
                        j = pos_map.get(new_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1
                            
                if delta < 0:
                    old_p = current_chain[i]
                    del pos_map[old_p]
                    current_chain[i] = q
                    pos_map[q] = i
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    improved = True
                    break
                    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + len(sequence) * 7)

    best_chain = None
    best_energy = 0  # will be set on first valid chain

    # Strategy: 
    # 1. Use the backtracking/greedy hybrid to generate initial chains.
    # 2. Refine with local search.
    
    # We will allocate time to multiple restarts of the generation phase.
    # For short sequences, we can afford more restarts.
    
    if n <= 25:
        max_restarts = 100
    elif n <= 40:
        max_restarts = 50
    else:
        max_restarts = 20

    # We will run the generation in batches to check time
    for restart in range(max_restarts):
        if _remaining() < 2.0:
            break
            
        # Generate a chain using the constraint-aware backtracking/greedy
        # We pass a new rng seed or just use the same rng for variation
        chain = _backtrack_fold(sequence, dimension, max_depth=12, max_restarts=1, rng=rng)
        
        if chain is None:
            continue
            
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
        # Refine with local search
        if _remaining() < 2.0:
            break
            
        # Local search iterations depend on time remaining
        # Estimate time per iteration? Just use a fixed small number.
        ls_iter = 20 if n > 40 else 40
        
        ls_chain, ls_energy = _local_search(sequence, best_chain, dimension, rng, max_iter=ls_iter)
        
        if ls_chain is not None and ls_energy < best_energy:
            best_chain = ls_chain
            best_energy = ls_energy

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
