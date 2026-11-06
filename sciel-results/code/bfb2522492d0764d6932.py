import time
import copy
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse rows as bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Initial State
    # We represent a state as:
    # 'gates': list of [u, v]
    # 'wire_masks': dict {wire_id: mask}
    # 'mask_to_wire': dict {mask: wire_id}
    # 'row_wires': list of sets of wire_ids
    # 'next_wire_id': int
    
    def create_initial_state():
        wire_masks = {}
        mask_to_wire = {}
        gates = []
        next_wire_id = n + 1
        
        for j in range(n):
            mask = 1 << j
            wire_id = j
            wire_masks[wire_id] = mask
            mask_to_wire[mask] = wire_id
            
        zero_wire = n
        wire_masks[zero_wire] = 0
        mask_to_wire[0] = zero_wire
        
        row_wires = [set() for _ in range(m)]
        for i in range(m):
            mask = row_masks[i]
            if mask == 0:
                row_wires[i].add(zero_wire)
            else:
                for j in range(n):
                    if (mask >> j) & 1:
                        row_wires[i].add(j)
                        
        return {
            'gates': gates,
            'wire_masks': wire_masks,
            'mask_to_wire': mask_to_wire,
            'row_wires': row_wires,
            'next_wire_id': next_wire_id
        }

    # Beam Search Parameters
    BEAM_WIDTH = 5
    TOP_K = 20
    
    # Initial beam
    beam = [create_initial_state()]
    best_state = None
    best_score = float('inf')
    
    def get_score(state):
        # Sum of lengths of row_wires
        return sum(len(s) for s in state['row_wires'])
    
    # Update best
    s0 = get_score(beam[0])
    best_score = s0
    best_state = beam[0]
    
    iter_count = 0
    while time.time() < DEADLINE - 2.0:
        new_beam = []
        
        for state in beam:
            # Get wires
            wire_ids = list(state['wire_masks'].keys())
            # Filter out zero wire for pair generation? 
            # XOR with zero is identity, so (u, zero) -> u. No new wire.
            # So we only consider pairs of non-zero wires.
            active_wires = [w for w in wire_ids if w != n]
            
            if len(active_wires) < 2:
                continue
                
            # Calculate gains for all pairs
            candidates = []
            k = len(active_wires)
            for i in range(k):
                u = active_wires[i]
                for j in range(i + 1, k):
                    v = active_wires[j]
                    
                    mask_u = state['wire_masks'][u]
                    mask_v = state['wire_masks'][v]
                    new_mask = mask_u ^ mask_v
                    
                    if new_mask in state['mask_to_wire']:
                        continue
                        
                    # Calculate gain
                    # Number of rows containing both u and v
                    count = 0
                    for r in range(m):
                        if u in state['row_wires'][r] and v in state['row_wires'][r]:
                            count += 1
                            
                    if count > 0:
                        candidates.append((count, u, v, new_mask))
                        
            # Sort candidates by gain (descending)
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Take top K
            for count, u, v, new_mask in candidates[:TOP_K]:
                # Create new state
                new_state = {
                    'gates': state['gates'] + [[u, v]],
                    'wire_masks': dict(state['wire_masks']),
                    'mask_to_wire': dict(state['mask_to_wire']),
                    'row_wires': [set(s) for s in state['row_wires']],
                    'next_wire_id': state['next_wire_id']
                }
                
                new_wire_id = new_state['next_wire_id']
                new_state['next_wire_id'] += 1
                new_state['wire_masks'][new_wire_id] = new_mask
                new_state['mask_to_wire'][new_mask] = new_wire_id
                
                # Update rows
                for r in range(m):
                    if u in new_state['row_wires'][r] and v in new_state['row_wires'][r]:
                        new_state['row_wires'][r].discard(u)
                        new_state['row_wires'][r].discard(v)
                        new_state['row_wires'][r].add(new_wire_id)
                        
                new_score = get_score(new_state)
                new_beam.append((new_score, new_state))
                
        if not new_beam:
            break
            
        # Sort by score
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top BEAM_WIDTH
        beam = [state for score, state in new_beam[:BEAM_WIDTH]]
        
        # Update best
        if beam:
            current_best = get_score(beam[0])
            if current_best < best_score:
                best_score = current_best
                best_state = beam[0]
                
        iter_count += 1
        
    # Finalize the best_state
    state = best_state
    gates = state['gates']
    wire_masks = state['wire_masks']
    mask_to_wire = state['mask_to_wire']
    row_wires = state['row_wires']
    next_wire_id = state['next_wire_id']
    zero_wire = n
    
    outputs = [0] * m
    for i in range(m):
        if not row_wires[i]:
            outputs[i] = zero_wire
        elif len(row_wires[i]) == 1:
            outputs[i] = list(row_wires[i])[0]
        else:
            wires = list(row_wires[i])
            current_wires = wires
            while len(current_wires) > 1:
                new_current_wires = []
                for k in range(0, len(current_wires), 2):
                    if k + 1 < len(current_wires):
                        w1 = current_wires[k]
                        w2 = current_wires[k + 1]
                        m1 = wire_masks[w1]
                        m2 = wire_masks[w2]
                        nm = m1 ^ m2
                        if nm in mask_to_wire:
                            new_current_wires.append(mask_to_wire[nm])
                        else:
                            new_w = next_wire_id
                            next_wire_id += 1
                            gates.append([w1, w2])
                            wire_masks[new_w] = nm
                            mask_to_wire[nm] = new_w
                            new_current_wires.append(new_w)
                    else:
                        new_current_wires.append(current_wires[k])
                current_wires = new_current_wires
            outputs[i] = current_wires[0]

    return {'gates': gates, 'outputs': outputs}
