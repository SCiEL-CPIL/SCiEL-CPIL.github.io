# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute clockwise edge masks for each demand
    cw_masks = []
    for s, t, w in demands:
        mask = np.zeros(n, dtype=np.int8)
        if s == t:
            # Full ring? Not possible in this problem, but handle gracefully
            mask[:] = 1
        else:
            # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)
            # Edge e connects node e to (e+1) mod 10
            # So clockwise path from s to t uses edges: s, (s+1)%10, ..., (t-1)%10
            cur = s
            while cur != t:
                mask[cur] = 1
                cur = (cur + 1) % n
        cw_masks.append(mask)
    
    cw_masks = np.array(cw_masks, dtype=np.int8)  # shape (15, 10)
    weights = np.array([w for _, _, w in demands], dtype=np.float64)  # shape (15,)

    # Compute loads for a given routing vector (0=cw, 1=ccw)
    def compute_loads(routing_bits):
        """
        routing_bits: array of 0/1, shape (15,)
        Returns edge loads, shape (10,)
        """
        # For each demand, if routing_bit == 0, use cw_mask; else use (1 - cw_mask)
        # Load on edge e = sum over demands of weight_d * (edge_e in route_d)
        # cw_route_contribution = weights * cw_masks
        # ccw_route_contribution = weights * (1 - cw_masks)
        
        # Efficient: 
        # loads = sum_d weight_d * (1 - 2*routing_bit_d) * cw_mask_d + sum_d weight_d * (routing_bit_d) * (1 - cw_mask_d)
        # Actually simpler: 
        # If bit=0: contribution = weight * cw_mask
        # If bit=1: contribution = weight * (1 - cw_mask)
        
        # Let's use: loads = (weights[:, None] * cw_masks).sum(axis=0) + 
        #                     (weights[:, None] * (1 - 2*routing_bits[:, None]) * (1 - cw_masks)).sum(axis=0)
        # That's not right. Let me think again.
        
        # contribution_d = weight_d * (cw_mask_d if bit_d==0 else (1-cw_mask_d))
        # = weight_d * (cw_mask_d * (1-bit_d) + (1-cw_mask_d) * bit_d)
        # = weight_d * (cw_mask_d - cw_mask_d*bit_d + bit_d - cw_mask_d*bit_d)
        # = weight_d * (cw_mask_d + bit_d - 2*bit_d*cw_mask_d)
        
        # So loads = sum_d weight_d * (cw_mask_d + bit_d - 2*bit_d*cw_mask_d)
        # = sum_d weight_d * cw_mask_d + sum_d weight_d * bit_d - 2 * sum_d weight_d * bit_d * cw_mask_d
        
        # Even simpler: just compute directly
        # expanded = weights[:, None] * (cw_masks * (1 - routing_bits[:, None]) + (1 - cw_masks) * routing_bits[:, None])
        # loads = expanded.sum(axis=0)
        
        # Most efficient for small n:
        # base_cw = weights @ cw_masks  # load if all cw
        # diff = weights * (1 - 2*routing_bits)  # for each demand, +1 if cw, -1 if ccw... wait
        # Let's just do:
        # For each demand d:
        #   if bit_d == 0: add weight_d * cw_mask_d
        #   if bit_d == 1: add weight_d * (1 - cw_mask_d)
        
        # Vectorized:
        # route_mask_d = cw_mask_d if bit_d==0 else (1-cw_mask_d)
        # = cw_mask_d * (1-bit_d) + (1-cw_mask_d) * bit_d
        
        route_masks = cw_masks * (1 - routing_bits[:, None]) + (1 - cw_masks) * routing_bits[:, None]
        loads = (weights[:, None] * route_masks).sum(axis=0)
        return loads

    def max_load(routing_bits):
        loads = compute_loads(routing_bits)
        return loads.max()

    # Start with all clockwise
    best_bits = np.zeros(D, dtype=np.int8)
    best_ml = max_load(best_bits)
    
    start_time = time.time()
    deadline = start_time + 590  # 10 second safety margin
    
    # Local search: try flipping each bit, keep if it improves
    improved = True
    while improved:
        improved = False
        for i in range(D):
            if time.time() > deadline:
                break
            # Try flipping bit i
            new_bits = best_bits.copy()
            new_bits[i] = 1 - new_bits[i]
            ml = max_load(new_bits)
            if ml < best_ml - 1e-9:
                best_bits = new_bits
                best_ml = ml
                improved = True
        if time.time() > deadline:
            break
    
    # Second pass: try pairs of flips
    if time.time() < deadline:
        improved = True
        while improved:
            improved = False
            for i in range(D):
                if time.time() > deadline:
                    break
                for j in range(i+1, D):
                    if time.time() > deadline:
                        break
                    new_bits = best_bits.copy()
                    new_bits[i] = 1 - new_bits[i]
                    new_bits[j] = 1 - new_bits[j]
                    ml = max_load(new_bits)
                    if ml < best_ml - 1e-9:
                        best_bits = new_bits
                        best_ml = ml
                        improved = True
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Convert to float array: 0 for cw, 1 for ccw
    result = best_bits.astype(np.float64)
    return result

# EVOLVE-BLOCK-END
