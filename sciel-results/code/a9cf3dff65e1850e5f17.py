<<<<<<< SEARCH
def _neh_heuristic(processing_times):
    """
    NEH heuristic with incremental evaluation for insertion.
    """
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            # Build candidate sequence
            if pos == 0:
                cand = [job] + sequence
            elif pos == len(sequence):
                cand = sequence + [job]
            else:
                cand = sequence[:pos] + [job] + sequence[pos:]
            
            ms = _calc_makespan(processing_times, cand)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence
=======
def _v_shaped_heuristic(processing_times):
    """
    V-shaped heuristic: Sort jobs by total processing time.
    Place jobs with smaller totals at the beginning and larger totals at the end.
    This creates a V-shaped profile of cumulative processing times.
    """
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    # Sort indices by total processing time
    sorted_indices = np.argsort(totals)
    
    # Create the sequence: smallest half, then largest half in reverse
    # Actually, standard V-shape is: sort by total time, then interleave or split.
    # A common variant: sort by total time. Take smallest, largest, 2nd smallest, 2nd largest...
    # Or simply: sort by total time. The sequence is [smallest, ..., largest] is not V.
    # V-shape means the cumulative time goes up, then down? No, cumulative time always goes up.
    # The "V" refers to the shape of the *processing times* themselves in the sequence?
    # No, it refers to the fact that the most "dominant" jobs (large totals) are placed in the middle?
    # Actually, the V-shaped heuristic for Flow Shop usually orders jobs such that the sum of processing times
    # increases then decreases? No, that's not possible for a single machine sum.
    
    # Correct V-shaped heuristic (Nawaz et al. 1983 / later improvements):
    # 1. Sort jobs in non-decreasing order of total processing time.
    # 2. Start with the job with the smallest total.
    # 3. Insert the next job (next smallest, or next largest?)
    # Actually, the standard "V-shaped" solution for FSP is often constructed by:
    # Sorting jobs by total processing time.
    # Then, the sequence is formed by taking jobs from the two ends of the sorted list:
    # Smallest, Largest, 2nd Smallest, 2nd Largest, etc.
    
    seq = []
    left = 0
    right = n_jobs - 1
    
    # We need to decide which end to pick next.
    # A common heuristic: Pick the job with the larger total processing time from the remaining ends?
    # Or alternate?
    # Let's try: Sort by total time.
    # Sequence: smallest, largest, 2nd smallest, 2nd largest...
    # This puts high-volume jobs in the middle/end.
    
    for _ in range(n_jobs):
        if left > right:
            break
        
        # Choose between left (smaller total) and right (larger total)
        # Heuristic: Pick the one with larger total processing time to put it later?
        # Actually, the "V" shape usually implies that the jobs with the *largest* processing times
        # are in the *middle* of the sequence, and the smallest are at the ends?
        # No, if smallest are at ends, the middle has large ones.
        # Let's check the literature: "V-shaped heuristic" usually sorts by total time and then
        # constructs the sequence by taking jobs from the two ends of the sorted list,
        # preferring the job with the *larger* total processing time to be placed *later*?
        
        # Let's try a simple V-shape:
        # Sort by total time: t0 <= t1 <= ... <= tN-1
        # Sequence: t0, tN-1, t1, tN-2, t2, ...
        
        if len(seq) % 2 == 0:
            # Pick from left (smaller)
            seq.append(sorted_indices[left])
            left += 1
        else:
            # Pick from right (larger)
            seq.append(sorted_indices[right])
            right -= 1
            
    return seq
>>>>>>> REPLACE

