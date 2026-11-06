"""
HP lattice protein folding using Beam Search over partial self-avoiding walks.
This strategy maintains a population of the K lowest-energy partial conformations
at each step, pruning inferior paths to manage the combinatorial explosion.
"""
# EVOLVE-BLOCK-START
import time
import random
import heapq

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Safety margin: 210s limit, leave 10s buffer -> 200s usable
    return 200.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _fold_beam_search(sequence, dimension, beam_width=50, time_limit=10.0, rng=None):
    """
    Perform Beam Search for the given sequence.
    Returns the best full conformation found.
    """
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    # Initial state
    start_pos = (0, 0)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_h = {start_pos} if is_h[0] else set()
    initial_energy = 0
    
    beam = [(initial_energy, initial_chain, initial_occupied, initial_h)]
    
    start_time = time.time()
    
    for i in range(1, n):
        new_beam = []
        is_current_h = is_h[i]
        
        for state in beam:
            energy, chain, occupied, h_pos = state
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                
                new_energy = energy
                new_h_pos = h_pos
                new_occupied = occupied | {nb}
                
                if is_current_h:
                    new_h_pos = h_pos | {nb}
                    for q in _neighbours(nb):
                        if q in h_pos:
                            if q != pred_pos:
                                new_energy -= 1
                
                new_chain = chain + [nb]
                new_beam.append((new_energy, new_chain, new_occupied, new_h_pos))
                
        if not new_beam:
            break
            
        # Prune with randomization for ties
        # Sort by energy
        new_beam.sort(key=lambda x: x[0])
        
        if len(new_beam) > beam_width:
            cutoff_energy = new_beam[beam_width - 1][0]
            
            # Separate states into 'definitely keep' (energy < cutoff) and 'maybe keep' (energy == cutoff)
            keep = []
            maybe = []
            for state in new_beam:
                if state[0] < cutoff_energy:
                    keep.append(state)
                elif state[0] == cutoff_energy:
                    maybe.append(state)
                else:
                    break # Since sorted, rest are worse
            
            # Need to fill remaining slots from 'maybe'
            slots_needed = beam_width - len(keep)
            if slots_needed > 0:
                if rng:
                    rng.shuffle(maybe)
                else:
                    random.shuffle(maybe)
                keep.extend(maybe[:slots_needed])
            
            beam = keep
        else:
            beam = new_beam
            
        if time.time() - start_time > time_limit:
            break
            
    if not beam:
        return None
        
    # Pick the best state from the final beam
    best_state = min(beam, key=lambda x: x[0])
    chain = best_state[1]
    occupied = best_state[2]
    h_pos = best_state[3]
    
    # If we broke early, extend greedily
    if len(chain) < n:
        for j in range(len(chain), n):
            is_j_h = is_h[j]
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            best_nb = None
            best_e = float('inf')
            
            candidates = []
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                e = 0
                if is_j_h:
                    for q in _neighbours(nb):
                        if q in h_pos and q != pred_pos:
                            e -= 1
                candidates.append((e, nb))
                
            if candidates:
                candidates.sort(key=lambda x: x[0])
                # Pick best, or random among best if tie
                best_e_val = candidates[0][0]
                best_opts = [nb for e, nb in candidates if e == best_e_val]
                if rng:
                    best_nb = rng.choice(best_opts)
                else:
                    best_nb = random.choice(best_opts)
                    
                chain.append(best_nb)
                occupied.add(best_nb)
                if is_j_h:
                    h_pos.add(best_nb)
            else:
                # Dead end, return None to signal failure
                return None
                
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search.
    """
    _init_time()
    n = len(sequence)
    
    # Adjust beam width based on time remaining and sequence length
    # Longer sequences need more time, but also explode faster.
    # Short sequences can use wider beams.
    if n <= 20:
        base_beam_width = 300
        base_time_limit = 5.0
    elif n <= 50:
        base_beam_width = 150
        base_time_limit = 10.0
    else:
        base_beam_width = 80
        base_time_limit = 15.0
        
    best_chain = None
    best_energy = 0 # 0 is neutral, we want negative
    
    # Strategy: Run multiple beam searches with different randomizations to explore different topologies.
    # We also adaptively increase beam width if time allows.
    
    num_runs = 0
    max_runs = 10
    
    # First pass: standard run
    # We will do a few runs with different seeds to break ties differently
    
    for run in range(max_runs):
        if _time_remaining() < 10:
            break
            
        # Adjust parameters for later runs: maybe wider beam?
        # Or just different random seeds.
        # Let's try slightly different beam widths for diversity?
        # Actually, keeping beam width constant but changing tie-breaking is safer for consistency.
        # But we can also try a "greedy" run (beam_width=1) quickly to get a baseline, 
        # though beam search with width > 1 usually dominates.
        
        # Let's just use different seeds for the random tie-breaking.
        seed = 12345 + run * 7
        rng = random.Random(seed)
        
        # For the first run, use the base parameters.
        # For subsequent runs, we might tweak slightly?
        # Let's stick to the same beam width to ensure we are exploring the same "space" 
        # but with different tie-breaks.
        
        current_bw = base_beam_width
        current_tl = base_time_limit
        
        # If we have lots of time left, we can afford a wider beam in early runs?
        # Or save time for more runs?
        # Let's keep it simple: fixed params per length class, vary seed.
        
        chain = _fold_beam_search(sequence, dimension, current_bw, current_tl, rng)
        
        if chain is not None and len(chain) == n:
            # Validate and compute energy
            valid = True
            occupied = set()
            for i, p in enumerate(chain):
                if p in occupied:
                    valid = False
                    break
                occupied.add(p)
                if i > 0:
                    prev = chain[i-1]
                    if abs(p[0]-prev[0]) + abs(p[1]-prev[1]) != 1:
                        valid = False
                        break
            
            if valid:
                # Calculate energy
                is_h = [c == 'H' for c in sequence]
                energy = 0
                pos_to_idx = {p: i for i, p in enumerate(chain)}
                for i, p in enumerate(chain):
                    if is_h[i]:
                        for q in _neighbours(p):
                            j = pos_to_idx.get(q)
                            if j is not None and j > i + 1 and is_h[j]:
                                energy -= 1
                                
                if best_chain is None or energy < best_energy:
                    best_chain = chain
                    best_energy = energy
                num_runs += 1
            else:
                # Invalid chain, ignore
                pass
        else:
            # Failed to complete, ignore
            pass
            
    if best_chain is None:
        # Fallback: Straight line
        best_chain = [list(p) for p in [(i, 0) for i in range(n)]]
        
    return best_chain
# EVOLVE-BLOCK-END
