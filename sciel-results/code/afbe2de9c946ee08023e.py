import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # Global ID registry for vectors to speed up set operations
    vec_to_id = {zero_vec: 0}
    next_id = 1
    
    def get_id(vec):
        nonlocal next_id
        if vec in vec_to_id:
            return vec_to_id[vec]
        vec_to_id[vec] = next_id
        next_id += 1
        return vec_to_id[vec]

    # Helper to check if a sequence is valid and return its product IDs and prefix vectors
    def compute_state(seq):
        pvecs = [zero_vec]
        pset = set()
        for x in seq:
            v_x = vecs[x]
            last = pvecs[-1]
            v_new_list = [last[i] + v_x[i] for i in range(num_primes)]
            v_new = tuple(v_new_list)
            
            # Check conflicts
            valid = True
            new_ids = []
            for i in range(len(pvecs)):
                diff_list = [v_new[j] - pvecs[i][j] for j in range(num_primes)]
                diff_tup = tuple(diff_list)
                diff_id = get_id(diff_tup)
                if diff_id in pset:
                    valid = False
                    break
                new_ids.append(diff_id)
            
            if not valid:
                return None
            
            pvecs.append(v_new)
            pset.update(new_ids)
        
        return (pvecs, pset)

    # Strategy:
    # 1. Start with a strong seed: all primes.
    # 2. Use randomized insertion of composites to break local optima.
    # 3. Use beam search logic implicitly by keeping the best sequence found so far,
    #    but the "beam" is effectively the population of random restarts/insertions.
    #    Actually, the current program uses beam search over the *selection* of numbers.
    #    The inspiration suggests starting with primes and inserting composites.
    #    
    #    Let's combine:
    #    The current program iterates n from 2 to 128, deciding include/exclude.
    #    This is a DP/Beam approach.
    #    The inspiration suggests a constructive approach: start with valid set, add elements.
    #    
    #    Let's try a hybrid:
    #    Run the existing beam search but with a larger width and tighter time management.
    #    The current program scores 1.0261. The inspirations scored ~1.017.
    #    The current program is likely better because it's a more systematic search (beam) than pure random insertion?
    #    Or maybe the random insertion found a specific structure?
    #    
    #    Let's look at the "inspiration 2" strategy: "Starting with all primes... randomized insertion of composites".
    #    This is a different search space traversal.
    #    
    #    Let's implement a multi-start approach:
    #    1. Run the beam search (current program logic) for a fixed time or iterations.
    #    2. Run the randomized insertion method for the remaining time.
    #    3. Return the best result.
    
    best_seq = []
    
    # Phase 1: Beam Search (Current Program Logic)
    # We will run this for a portion of the time, say 30 seconds, to get a strong baseline.
    beam_start = time.time()
    beam_deadline = beam_start + 30.0
    
    beam = [([], [zero_vec], set())]
    beam_width = 500 # Increase width since we have more time per phase? Or keep high?
    
    # Optimization: Pre-compute vector sums? No, vectors are high-dim.
    # The bottleneck is the set union and the inner loop over pvecs.
    
    for n in range(2, 129):
        if time.time() > beam_deadline:
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for seq, pvecs, pset in beam:
            # Option 1: Exclude n
            new_beam.append((seq, pvecs, pset))
            
            # Option 2: Include n
            last_pvec = pvecs[-1]
            v_new_list = [last_pvec[i] + v_n[i] for i in range(num_primes)]
            v_new = tuple(v_new_list)
            
            valid = True
            new_prod_ids = []
            
            # Check all suffixes ending at n
            for i in range(len(pvecs)):
                diff_list = [v_new[j] - pvecs[i][j] for j in range(num_primes)]
                diff_tup = tuple(diff_list)
                diff_id = get_id(diff_tup)
                
                if diff_id in pset:
                    valid = False
                    break
                
                new_prod_ids.append(diff_id)
            
            if valid:
                new_seq = seq + [n]
                new_pvecs = pvecs + [v_new]
                new_pset = pset.union(new_prod_ids)
                new_beam.append((new_seq, new_pvecs, new_pset))
        
        # Select top beam_width states
        # Sort by length descending
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        beam = new_beam[:beam_width]
        
        if beam and len(beam[0][0]) > len(best_seq):
            best_seq = beam[0][0]
            
        if beam and len(beam[0][0]) == 127:
            return beam[0][0]

    # Phase 2: Randomized Insertion (Inspiration Logic)
    # Start with all primes, then try to insert composites.
    # Or start with the best_seq from Phase 1 and try to improve?
    # The inspiration says "Starting with all primes".
    # Let's try both: start with primes, and start with best_seq.
    
    # Reset random seed for reproducibility? No, let it be random.
    
    def try_improve(seq):
        # Try to insert composites into seq
        # Sort seq
        seq = sorted(seq)
        # Compute state
        state = compute_state(seq)
        if state is None:
            return seq
        pvecs, pset = state
        
        # Candidate composites: [2..128] \ seq
        candidates = [x for x in range(2, 129) if x not in seq]
        random.shuffle(candidates)
        
        improved = False
        for c in candidates:
            # Try inserting c into every possible position in seq
            # To be efficient, we can just try appending? No, insertion can be anywhere.
            # But the sequence must be strictly increasing.
            # So we can only insert c at a position where prev < c < next.
            
            # Find valid insertion positions
            # seq is sorted.
            # We can binary search for c's position.
            # If c is already in seq, skip. (We checked)
            
            # Since we are iterating candidates, and seq changes, this is O(N^2) roughly.
            # Let's just try to insert c into the *end* first? No, that's greedy.
            # Let's try all positions.
            
            # Optimization: Only try positions where c fits.
            # Since seq is sorted, find index i such that seq[i-1] < c < seq[i]
            # There is at most one such position?
            # Yes, because seq is strictly increasing.
            # So c can only be inserted at one specific index to maintain sorted order.
            
            # Find insertion index
            lo, hi = 0, len(seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if seq[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            
            ins_idx = lo
            
            # Check bounds
            if ins_idx < len(seq) and seq[ins_idx] == c:
                continue # Should not happen
            
            # Construct new_seq
            new_seq = seq[:ins_idx] + [c] + seq[ins_idx:]
            
            # Check validity
            new_state = compute_state(new_seq)
            if new_state is not None:
                seq = new_seq
                pvecs, pset = new_state
                improved = True
        
        return seq

    # Run randomized insertion for remaining time
    ins_start = time.time()
    ins_deadline = start_time + 85.0
    
    # Initialize candidates for insertion
    # Start with all primes
    primes_seq = [p for p in primes if p <= 128]
    
    # Also try to improve the best_seq from beam search
    # We will alternate between improving primes_seq and best_seq
    
    current_best = best_seq
    if len(primes_seq) > len(current_best):
        current_best = primes_seq
    
    # Create a list of "tracks" to improve
    # Track 1: Start with primes
    # Track 2: Start with beam best
    
    tracks = []
    if primes_seq:
        tracks.append(list(primes_seq))
    if best_seq:
        tracks.append(list(best_seq))
    
    # If tracks are empty, return empty
    if not tracks:
        return []
    
    # Shuffle tracks? No, process them.
    # We will iterate through the tracks, trying to improve them.
    # To avoid bias, we can shuffle the order of candidate composites in each call.
    
    # Actually, the `try_improve` function shuffles candidates internally.
    # So we can just call it multiple times on the same track?
    # No, once a composite is inserted, it's in the seq.
    # The function iterates through ALL candidates not in seq.
    # So one call to try_improve is a full pass.
    
    # We can run multiple passes.
    
    while time.time() < ins_deadline:
        for i, track in enumerate(tracks):
            if time.time() >= ins_deadline:
                break
            new_track = try_improve(track)
            tracks[i] = new_track
            if len(new_track) > len(current_best):
                current_best = new_track
        
        # If we made no progress in a full round of tracks, we might be stuck.
        # But try_improve shuffles, so it might find different insertions?
        # No, try_improve is deterministic given the seq and the shuffle.
        # If it returns the same seq, we are stuck.
        
        # Let's add a random restart:
        # Pick a random subset of the current best?
        # Or just start a new track from a random valid sequence?
        
        # Simpler: If we have time, start a new track from a random permutation of primes + some composites?
        # Let's just add a new track from the current best, but shuffled?
        # No, seq must be sorted.
        
        # Let's try a different strategy for Phase 2:
        # Randomly remove elements from the current best to create "holes", then re-insert?
        # This is complex.
        
        # Let's just rely on the multiple passes.
        # If no improvement is made in a pass, we can break.
        
        # How to detect no improvement?
        # try_improve returns the improved seq.
        # We can compare lengths.
        
        improved_any = False
        # We need to track if any track improved.
        # Let's modify the loop.
        
        # Actually, the loop above already updates tracks.
        # Let's check if any track length increased in the last round.
        # We didn't store previous lengths.
        
        # Let's simplify:
        # Just run try_improve on a single "best" track repeatedly?
        # No, the shuffle means it might find different insertions in different runs?
        # No, try_improve iterates through candidates in random order.
        # If insertion at position X fails, it doesn't retry X with a different context?
        # Wait, if we insert A, then B might become insertable.
        # So multiple passes on the same track are useful.
        
        # But if we insert all possible composites, we reach a local optimum.
        # To escape, we need to remove some.
        
        # Let's add a "perturbation" step:
        # If stuck, remove a random composite from the track and re-run.
        
        # For now, let's just run the loop.
        
        # Check if we should break early if no progress?
        # Let's track the best length found in Phase 2.
        
    return current_best

# Run a quick test to ensure it doesn't crash
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
