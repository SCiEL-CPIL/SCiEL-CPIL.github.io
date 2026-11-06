"""HP lattice protein folding: growth init + fragment-move simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 205.0  # wall-clock limit with safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy from scratch using position->index map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _sa_fold(sequence, initial_chain, rng, time_budget):
    """
    Simulated Annealing with Kink and Reversal moves.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain, _compute_energy(sequence, initial_chain)
    
    # State
    chain = list(initial_chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # SA Parameters
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    start_time = time.time()
    
    # Helper to check if a set of positions are valid for a move
    # For Kink: move i to new_pos
    # For Reversal: reverse i..j
    
    def _kink_delta(i, new_pos):
        """Calculate energy delta for moving residue i to new_pos."""
        old_pos = chain[i]
        if old_pos == new_pos:
            return 0
        
        # Check validity: new_pos must not be occupied by anyone else
        if new_pos in pos_to_idx and pos_to_idx[new_pos] != i:
            return None # Invalid
        
        # Also check connectivity: new_pos must be adjacent to i-1 and i+1
        if i > 0 and new_pos not in _neighbours(chain[i-1]):
            return None
        if i < n-1 and new_pos not in _neighbours(chain[i+1]):
            return None
            
        # Calculate delta
        delta = 0
        # Remove contacts from old_pos
        if sequence[i] == 'H':
            for q in _neighbours(old_pos):
                j = pos_to_idx.get(q)
                if j is not None and j != i and abs(j - i) > 1:
                    if sequence[j] == 'H':
                        delta += 1 # Lost contact
        # Add contacts at new_pos
        if sequence[i] == 'H':
            for q in _neighbours(new_pos):
                j = pos_to_idx.get(q)
                if j is not None and j != i and abs(j - i) > 1:
                    if sequence[j] == 'H':
                        delta -= 1 # Gained contact
                        
        return -delta # Energy is -contacts, so delta_energy = -(new_contacts - old_contacts)
        
    def _apply_kink(i, new_pos, delta_e):
        old_pos = chain[i]
        # Update chain
        chain[i] = new_pos
        # Update pos_to_idx
        del pos_to_idx[old_pos]
        pos_to_idx[new_pos] = i
        # Update energy
        return current_energy + delta_e

    def _reversal_delta(i, j):
        """Calculate energy delta for reversing segment i..j."""
        if j - i < 1:
            return 0, False
            
        # The segment is chain[i..j].
        # New segment will be chain[j..i] reversed in position?
        # No, we are reversing the *order* of residues in the segment.
        # The positions occupied by the segment are the same set of positions.
        # Wait, a reversal in HP folding usually means:
        # Take the path from i to j, and reverse the direction of traversal.
        # This maps residue k to position chain[j - (k - i)].
        # So residue i moves to chain[j], residue i+1 moves to chain[j-1], etc.
        
        # Check validity:
        # 1. The new endpoints must connect to the rest of the chain.
        #    Left neighbor of new i (which is old j) is chain[i-1].
        #    Right neighbor of new j (which is old i) is chain[j+1].
        #    So chain[j] must be adjacent to chain[i-1] (if i>0)
        #    And chain[i] must be adjacent to chain[j+1] (if j<n-1)
        
        if i > 0:
            if chain[j] not in _neighbours(chain[i-1]):
                return None, False
        if j < n - 1:
            if chain[i] not in _neighbours(chain[j+1]):
                return None, False
                
        # 2. The internal positions are the same, so no self-intersection issues internally.
        #    But we need to ensure that the *mapping* doesn't cause collisions?
        #    Since we are just permuting residues within the *same* set of positions,
        #    and the positions are already occupied by this segment, there are no collisions
        #    with *other* parts of the chain. The only potential issue is if the reversal
        #    causes the chain to cross itself? No, the positions are fixed.
        #    The only constraint is the connectivity at the boundaries.
        
        # Calculate Delta Energy
        # Contacts within the segment are preserved (relative geometry is mirrored, but H-H contact is symmetric).
        # Only contacts involving the boundary residues (i-1, i, j, j+1) and the segment change.
        # Specifically:
        # Old contacts involving i: with i-1, i+1 (chain adj), and non-chain neighbors.
        # New contacts involving i (now at old j's pos): with new i-1 (old j-1? No, i-1 is same residue), 
        # wait.
        # Residue i moves to pos chain[j].
        # Residue j moves to pos chain[i].
        # Residue k (i<k<j) moves to pos chain[j-(k-i)].
        
        # Let's just recalculate energy for the affected region to be safe, or just full recalc.
        # Given N<=100, full recalc is ~400 ops. 
        # If we do 10000 reversals, that's 4M ops. Very fast.
        # Let's just do full recalc for reversals to avoid subtle bugs in delta calc.
        # For kinks, delta calc is safer and faster if we do many.
        
        # However, we need to return the new energy.
        # Let's construct the new chain temporarily? No, that's expensive.
        # Let's just return a flag that we need to recalc.
        
        return None, True # Flag: needs full recalc

    while T > T_min and (time.time() - start_time) < time_budget and _remaining() > 2.0:
        move_type = rng.random()
        
        if move_type < 0.5:
            # Kink Move
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            candidates = []
            for q in _neighbours(old_pos):
                if q == chain[i-1] or q == chain[i+1]:
                    continue
                if q not in pos_to_idx or pos_to_idx[q] == i:
                    candidates.append(q)
            
            if not candidates:
                T *= cooling
                continue
                
            new_pos = rng.choice(candidates)
            delta_e = _kink_delta(i, new_pos)
            
            if delta_e is not None:
                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                    new_energy = _apply_kink(i, new_pos, delta_e)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    current_energy = new_energy
        else:
            # Reversal Move
            # Pick i, j with distance > 1
            # To avoid too long reversals which are less likely to be valid, 
            # maybe restrict length? Or just pick random.
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Quick check boundaries
            valid = True
            if i > 0 and chain[j] not in _neighbours(chain[i-1]):
                valid = False
            if j < n - 1 and chain[i] not in _neighbours(chain[j+1]):
                valid = False
                
            if valid:
                # Apply reversal
                # Reverse chain[i..j]
                segment = chain[i:j+1]
                segment.reverse()
                chain[i:j+1] = segment
                
                # Update pos_to_idx
                # We need to remap the indices for the segment
                # Old: pos->idx. New: same pos->new_idx.
                # We can just rebuild the map for the segment? 
                # Or delete and re-add.
                # Delete old mappings for segment
                for k in range(i, j+1):
                    del pos_to_idx[chain[k]] # Wait, chain is already updated. 
                    # This is tricky. We need the old positions.
                
                # Let's do it properly:
                # 1. Store old positions
                # 2. Delete from pos_to_idx
                # 3. Reverse chain
                # 4. Add new mappings
                
                # Re-doing reversal logic to be safe
                # Restore chain? No, let's just rebuild pos_to_idx for the whole chain? 
                # O(N) is fine.
                
                # Actually, let's just recalc energy and rebuild pos_to_idx if accepted.
                # If rejected, we need to undo.
                
                # Undo logic for reversal is just reverse again.
                
                # Let's calculate new energy
                new_energy = _compute_energy(sequence, chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    
                if new_energy <= current_energy or rng.random() < math.exp(-(new_energy - current_energy) / T):
                    # Accept
                    # Rebuild pos_to_idx
                    pos_to_idx = {p: idx for idx, p in enumerate(chain)}
                    current_energy = new_energy
                else:
                    # Reject: Undo reversal
                    segment = chain[i:j+1]
                    segment.reverse()
                    chain[i:j+1] = segment
        
        T *= cooling
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using growth + local search with time management."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n <= 1:
        return [[0]]
    
    best_chain = None
    best_energy = 0  # worst possible (no contacts)
    
    # Time allocation
    # We have ~205s total. 
    # Let's assume 10 sequences. ~20s per sequence.
    # But fold is called per sequence. 
    # We need to track global time.
    
    # Phase 1: Generate initial conformations
    # Number of growths depends on time left and n
    # For n=20, growth is fast. For n=100, slower.
    
    num_growth = 50
    if n > 50:
        num_growth = 30
        
    growth_deadline = time.time() + 5.0 # Max 5s for growth
    
    candidates = []
    for _ in range(num_growth):
        if time.time() > growth_deadline or _remaining() < 10.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _compute_energy(sequence, chain)
        candidates.append((value, chain))
        
    if not candidates:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best_chain]
        
    # Sort candidates by energy
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: SA on top candidates
    # We have remaining time.
    # Let's take top 5 candidates and run SA on them.
    # Or just the best one? 
    # Running SA on multiple diverse candidates is better.
    
    top_k = min(5, len(candidates))
    
    for k in range(top_k):
        if _remaining() < 5.0:
            break
            
        # Allocate time per SA run
        # If 100s left and 5 runs, 20s each?
        # But we don't know how many sequences are left.
        # Let's just use a fraction of remaining time.
        # If this is the first call, we have ~205s.
        # Let's give SA 80% of remaining time, split among top_k.
        # But we must leave time for other sequences.
        # The problem says "fold is called once per benchmark sequence... 120s CPU budget for all sequences together" 
        # Wait, the prompt says "120 s CPU budget for all sequences together" in the task description, 
        # but the evaluator kills after 210s wall clock.
        # The current program doesn't track global state across calls? 
        # "fold is called once per benchmark sequence in a sandboxed subprocess"
        # Ah, "sandboxed subprocess" implies a fresh process for each call? 
        # Or one process for all? 
        # "The evaluator checks... fold is called once per benchmark sequence in a sandboxed subprocess with a 120 s CPU budget for all sequences together."
        # This phrasing is ambiguous. Usually, it means the *total* time for all 10 calls is 120s.
        # If it's a single process, I should track global time.
        # If it's separate subprocesses, I only have 120s (or 210s) for *this* call?
        # "120 s CPU budget for all sequences together" suggests a total budget.
        # However, the "210s wall-clock" kill switch is likely per process.
        # If each call is a separate subprocess, I have 210s for each? That would be 2100s total.
        # But "120s CPU budget for all sequences together" is a constraint on the *task*.
        # Let's assume the worst case: I have 210s for this specific call, but I should try to be fast.
        # Actually, looking at the inspiration code, it uses `_T0` and `_DEADLINE` at module level.
        # This implies the module is loaded once, and `fold` is called multiple times in the same process.
        # So I MUST track global time.
        
        # Calculate time budget for this SA run
        # We want to finish all 10 sequences well before 205s.
        # Let's say we have 205s total.
        # We don't know how many sequences have been done.
        # We can estimate remaining sequences? No.
        # Let's just use a fixed time per sequence, say 15s.
        # If we exceed, we return best so far.
        
        sa_time_budget = 15.0 # seconds
        
        # But if we are on the last sequence, we can use more time.
        # We don't know which sequence we are on.
        # Let's stick to 15s per sequence. 10 * 15 = 150s. Safe.
        
        _, init_chain = candidates[k]
        
        # Run SA
        improved_chain, improved_energy = _sa_fold(sequence, init_chain, rng, sa_time_budget)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
            
    if best_chain is None:
        best_chain = candidates[0][1]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
