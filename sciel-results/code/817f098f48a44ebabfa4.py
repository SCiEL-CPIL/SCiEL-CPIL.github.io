<<<<<<< SEARCH
def _local_improve(placements, boxes, container, deadline, rng, max_iters=50):
    """Local search: remove and re-insert boxes to improve packing."""
    W, H, D = container
    n = len(boxes)
    
    placed_list = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed_list.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    improved = False
    for _ in range(min(max_iters, len(loaded_indices) * 2)):
        if time.time() > deadline:
            break
        idx = rng.choice(loaded_indices)
        p = placements[idx]
        box_entry = (p[1], p[2], p[3], p[4], p[5], p[6])
        
        # Remove
        try:
            placed_list.remove(box_entry)
        except ValueError:
            placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            continue
        
        # Try to re-place
        if _place_box(placed_list, boxes, idx, container, deadline, 'extreme', 'compact'):
            new_p = placed_list[-1]
            placements[idx] = (0, new_p[0], new_p[1], new_p[2], new_p[3], new_p[4], new_p[5])
            improved = True
        else:
            # Put it back
            placed_list.append(box_entry)
    
    total_vol = sum(placements[i][4] * placements[i][5] * placements[i][6] for i in range(n) if placements[i][0] == 0)
    return placements, total_vol
=======
def _local_improve(placements, boxes, container, deadline, rng, max_iters=100):
    """Local search: remove and re-insert boxes to improve packing. Also tries to insert unplaced boxes."""
    W, H, D = container
    n = len(boxes)
    
    placed_list = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed_list.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    unplaced_indices = [i for i in range(n) if placements[i][0] == -1]
    
    if not loaded_indices and not unplaced_indices:
        return placements, 0.0
    
    for _ in range(min(max_iters, len(loaded_indices) * 3 + len(unplaced_indices) * 2)):
        if time.time() > deadline:
            break
        
        # Decide action: re-place loaded box or try to place unplaced box
        if unplaced_indices and (not loaded_indices or rng.random() < 0.3):
            # Try to place an unplaced box
            idx = rng.choice(unplaced_indices)
            if _place_box(placed_list, boxes, idx, container, deadline, 'extreme', 'compact'):
                new_p = placed_list[-1]
                placements[idx] = (0, new_p[0], new_p[1], new_p[2], new_p[3], new_p[4], new_p[5])
                unplaced_indices.remove(idx)
                loaded_indices.append(idx)
        elif loaded_indices:
            # Remove and re-place a loaded box
            idx = rng.choice(loaded_indices)
            p = placements[idx]
            box_entry = (p[1], p[2], p[3], p[4], p[5], p[6])
            
            # Remove
            try:
                placed_list.remove(box_entry)
            except ValueError:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                loaded_indices.remove(idx)
                unplaced_indices.append(idx)
                continue
            
            # Try to re-place
            if _place_box(placed_list, boxes, idx, container, deadline, 'extreme', 'compact'):
                new_p = placed_list[-1]
                placements[idx] = (0, new_p[0], new_p[1], new_p[2], new_p[3], new_p[4], new_p[5])
            else:
                # Put it back
                placed_list.append(box_entry)
    
    total_vol = sum(placements[i][4] * placements[i][5] * placements[i][6] for i in range(n) if placements[i][0] == 0)
    return placements, total_vol
>>>>>>> REPLACE

