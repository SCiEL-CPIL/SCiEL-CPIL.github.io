"""HP lattice protein folding: beam search with local refinement."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_full(seq, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and j > i + 1 and seq[j] == "H":
                contacts += 1
    return -contacts


def _beam_search(seq, beam_width=50):
    """Beam search to find a good initial conformation."""
    n = len(seq)
    if n <= 2:
        return [(0, 0), (1, 0)] if n == 2 else [(0, 0)]
    
    # Each beam entry: (energy, chain_list, occupied_set)
    # We use a list of (energy, chain) and maintain occupied sets separately
    beams = [(-1, [(0, 0)], {(0, 0): 0})]
    
    for i in range(1, n):
        new_beams = []
        for energy, chain, occupied in beams:
            last = chain[-1]
            x, y = last
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (x + dx, y + dy)
                if new_pos in occupied:
                    continue
                new_chain = chain + [new_pos]
                new_occupied = occupied.copy()
                new_occupied[new_pos] = i
                
                # Calculate new energy contribution
                new_energy = energy
                if seq[i] == "H":
                    for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = new_occupied.get((new_pos[0] + dx2, new_pos[1] + dy2))
                        if j is not None and j < i - 1 and seq[j] == "H":
                            new_energy -= 1
                
                new_beams.append((new_energy, new_chain, new_occupied))
        
        if not new_beams:
            # No valid extensions, return best we had
            return beams[0][1]
        
        # Sort by energy and keep top beam_width
        new_beams.sort(key=lambda x: x[0])
        beams = new_beams[:beam_width]
    
    return beams[0][1]


def _local_refine(seq, chain, max_iterations=1000):
    """Local search using 2-opt and single-residue moves."""
    n = len(seq)
    if n <= 2:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy_full(seq, best_chain)
    
    for _ in range(max_iterations):
        if _remaining() < 5.0:
            break
        
        improved = False
        
        # Try single-residue moves: move one residue to an adjacent empty spot
        # and check if the chain remains self-avoiding
        for i in range(1, n - 1):
            x, y = best_chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (x + dx, y + dy)
                # Check if new_pos is free (not occupied by any other residue)
                is_free = True
                for j in range(n):
                    if j != i and best_chain[j] == new_pos:
                        is_free = False
                        break
                if not is_free:
                    continue
                
                # Also check that the move doesn't create adjacency violations
                # The chain must still be a valid walk: consecutive residues must be adjacent
                # If we move residue i, we need to check that residue i-1 and i+1 are still
                # adjacent to the new position... but they're not necessarily.
                # Actually, moving a single residue breaks the chain connectivity.
                # We need a different approach.
                pass
        
        # Try 2-opt: reverse a segment of the chain
        for i in range(1, n - 2):
            for j in range(i + 2, n):
                if j - i < 2:
                    continue
                # Reverse segment [i, j]
                new_chain = best_chain[:i] + best_chain[i:j+1][::-1] + best_chain[j+1:]
                
                # Check self-avoidance
                if len(set(new_chain)) != n:
                    continue
                
                # Check unit steps
                valid = True
                for k in range(n - 1):
                    x1, y1 = new_chain[k]
                    x2, y2 = new_chain[k+1]
                    if abs(x1 - x2) + abs(y1 - y2) != 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_energy = _energy_full(seq, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    improved = True
        
        if not improved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Fold a protein sequence into a low-energy conformation."""
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0
    
    # Try multiple beam widths for diversity
    for beam_width in [20, 50, 100]:
        if _remaining() < 10.0:
            break
        
        chain = _beam_search(sequence, beam_width)
        if chain is None or len(chain) != n:
            continue
        
        # Local refinement
        refined = _local_refine(sequence, chain)
        energy = _energy_full(sequence, refined)
        
        if best_chain is None or energy < best_energy:
            best_chain = refined
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
