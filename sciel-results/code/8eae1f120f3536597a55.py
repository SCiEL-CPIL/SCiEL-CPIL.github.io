import numpy as np
import time
import random

N = 18

def construct():
    """
    Constructs a sorting network using genetic algorithm with constructive heuristics
    and greedy removal.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Generate Seed Networks
    # Start with a few known good networks or generated ones
    # We will use the Odd-Even Transposition network as the primary seed,
    # and maybe a few mutated versions.
    
    def generate_oo_network():
        """Generates the standard Odd-Even Transposition network."""
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))
        return network

    # We can also try to generate a random valid network via random comparators and then sort?
    # No, that doesn't guarantee sorting.
    # So we stick to OET as the base.
    
    initial_network = generate_oo_network()
    
    # 2. Prepare Verification State
    size = 1 << N
    k_range = np.arange(size, dtype=np.uint64)
    initial_states = np.empty((N, size), dtype=np.uint8)
    for j in range(N):
        initial_states[j] = ((k_range >> j) & 1).astype(np.uint8)

    def is_sorted(states):
        # Check if states[0] >= states[1] ...
        # Since binary, states[j] >= states[j+1] is equivalent to NOT (states[j]=0 AND states[j+1]=1)
        # Vectorized check: np.any(states[:-1] < states[1:])
        return not np.any(states[:-1] < states[1:])

    def apply_network(states, network):
        """Applies a network of comparators to states (in-place)."""
        for i, j in network:
            si = states[i].copy()
            sj = states[j].copy()
            states[i] = si & sj
            states[j] = si | sj
        return states

    def try_remove_comparators(network):
        """
        Attempts to remove redundant comparators from the network greedily.
        Returns the reduced network.
        """
        # Make a copy to avoid modifying the original if we fail completely?
        # We mutate the copy.
        net = network[:]
        
        # Initialize state
        states = initial_states.copy()
        
        # Remove invalid or unnecessary leading comparators?
        # First, ensure the network is valid.
        apply_network(states, net)
        if not is_sorted(states):
            return None # Invalid network
        
        # Greedy removal pass
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Iterate backwards
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline:
                    return net
                
                i, j = net[idx]
                
                # Save state of wires i and j
                si = states[i].copy()
                sj = states[j].copy()
                
                # Remove comparator from current state
                # Since we can't "un-apply" easily, we reconstruct state for this step?
                # No, we stored the state *before* applying the rest of the network in the previous step?
                # No, `states` currently holds the result of applying `net[0:idx]` to initial_states.
                # If we remove `net[idx]`, the new state for the remaining network would be:
                # apply `net[0:idx]` then `net[idx+1:]`.
                # Current `states` is apply(`net[0:idx]`) then apply(`net[idx]`).
                # To simulate removing `net[idx]`, we need the state *before* `net[idx]`.
                # Which is `si`, `sj` and the rest of the wires.
                # So:
                states[i] = si
                states[j] = sj
                
                # Now apply the rest of the network: net[idx+1:]
                for k_idx in range(idx + 1, len(net)):
                    ki, kj = net[k_idx]
                    ski = states[ki].copy()
                    skj = states[kj].copy()
                    states[ki] = ski & skj
                    states[kj] = ski | skj
                
                # Check if sorted
                if is_sorted(states):
                    # Removal is valid
                    net.pop(idx)
                    changed = True
                    # The `states` variable is now the correct state for the reduced network.
                    # We continue with this state.
                else:
                    # Removal failed, revert state to include net[idx]
                    # Re-apply net[idx]
                    states[i] = si & sj
                    states[j] = si | sj
                    # Re-apply net[idx+1:] ?
                    # Wait, the state `states` was derived from `states_before_idx` (with si, sj restored)
                    # by applying `net[idx+1:]`.
                    # So `states` currently holds the result of NOT applying `net[idx]` but applying the rest.
                    # To revert to the state WITH `net[idx]`, we need to:
                    # 1. Restore `states` to `states_before_idx` (where i=si, j=sj, others same)
                    # 2. Apply `net[idx]`
                    # 3. Apply `net[idx+1:]`
                    # This is expensive.
                    # Alternative: Keep a snapshot of `states` before the trial?
                    # That would be O(N * 2^N) memory per step, or just copy the whole array.
                    # Let's copy the whole state array for safety and simplicity, as 4.7MB copy is fast enough?
                    # 4.7MB copy is ~1ms. With 1000 steps, that's 1s. Acceptable.
                    
                    # Actually, let's just re-apply from scratch for the rest?
                    # No, let's optimize the revert logic.
                    # We have:
                    # S_prev (with si, sj)
                    # S_trial = Apply(S_prev, net[idx+1:])
                    # S_current (with net[idx]) = Apply(S_prev, [net[idx]] + net[idx+1:])
                    
                    # To get back to S_current from S_trial:
                    # We can't easily.
                    # So we must save S_prev? Or S_current?
                    # Let's save S_current before the trial.
                    
                    pass # This logic is getting complex. Let's use full state copy for reliability.
        
        return net

    # Simplified try_remove_comparators with full state copy for revert
    def try_remove_comparators_v2(network):
        net = network[:]
        states = initial_states.copy()
        
        # Verify full network first
        apply_network(states, net)
        if not is_sorted(states):
            return None

        # We maintain `states` as the result of applying `net[0:current_end]` to initial_states?
        # No, let's maintain `states` as the result of applying the *current* `net` to initial_states.
        # When we try to remove `net[idx]`, we:
        # 1. Save a copy of the current `states` (call it `states_full`).
        # 2. Compute the state if `net[idx]` were removed.
        #    This requires applying `net[0:idx]` to `initial_states`, then `net[idx+1:]`.
        #    This is O(idx + len - idx - 1) steps.
        # 3. If valid, update `net` and recompute `states`?
        #    Or just keep the new state.
        
        # This recomputation is expensive.
        
        # Better approach:
        # Maintain `states_after_k` for k=0..len(net)?
        # states_after_0 = initial_states
        # states_after_{k+1} = apply(states_after_k, net[k])
        # To remove net[idx], we need to check if:
        # apply(states_after_{idx+1}, net[idx+1:]) is sorted.
        # Note: states_after_{idx+1} = apply(states_after_idx, net[idx]).
        
        # So we can pre-compute states_after_k for all k.
        # Memory: (len(net) + 1) * N * 2^N * 1 byte.
        # For 100 comparators: 100 * 4.7MB = 470MB. Might be too much.
        # But we can compute them on the fly backwards?
        
        # Let's stick to the simple "try and reapply" method but optimize the reapply.
        # When removing net[idx]:
        # 1. Save states (4.7MB copy)
        # 2. Apply net[0:idx] to initial_states -> S_mid
        # 3. Apply net[idx+1:] to S_mid -> S_new
        # 4. Check S_new
        # 5. If fail, restore states.
        
        # Cost of step 2: O(idx * 2^N * N) vector ops.
        # Cost of step 3: O((len-idx-1) * 2^N * N) vector ops.
        # Total O(len * 2^N * N).
        # 153 * 262144 * 18 ~ 700M ops.
        # Vectorized: ~700M / 100M (np throughput) ~ 7s per check.
        # If we do 100 checks, that's 700s. Too slow.
        
        # We need a faster way.
        
        # Back to the "local state" trick.
        # If we iterate backwards, we can maintain `states` as the result of applying the *remaining* network to `initial_states`?
        # No.
        
        # Let's try a different strategy:
        # Only remove if it's at the END of the network.
        # If we only remove from the end, we just need to check if the network *without the last comp* is valid.
        # State = apply(net[0:-1], initial). Check sorted.
        # If yes, pop. State remains valid for the new net?
        # No, the state is the *output* of the net.
        # If we pop the last one, the new output is different.
        # But we already have the output of the *prefix* (which is the new full network).
        # So:
        # 1. Compute S_prefix = apply(net[0:-1], initial)
        # 2. Check if S_prefix is sorted.
        # 3. If yes, pop. Set states = S_prefix.
        # 4. If no, keep.
        
        # This only allows removing from the end.
        # Can we remove from anywhere?
        # If we remove a middle comparator, we break the state chain.
        
        # Strategy:
        # 1. Start with full net.
        # 2. Try to remove from the end.
        # 3. If end is stuck, try to move comparators to the end? (Swap)
        # 4. Or, try removing any, but accept the cost of reapplication.
        #    Given time constraints, let's stick to removal from the end first.
        #    This often works well because transposition networks have redundancy at the end.
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Only try to remove from the end for now
            if len(net) > 0:
                # Compute state for net[0:-1]
                # We don't have this state stored if we haven't just computed it.
                # So we must recompute.
                # Recompute cost: O(len * 2^N * N).
                # This is expensive.
                
                # Better: Maintain states_after_k?
                # Let's pre-compute states_after_k array.
                # List of states.
                # states_after[0] = initial
                # states_after[k] = apply(states_after[k-1], net[k-1])
                
                # This uses O(len * 4.7MB) memory.
                # For 153 comparators: ~700MB.
                # Might be OK on a modern machine, but risky.
                
                # Alternative:
                # Just do a full verification for the whole network whenever we make a change.
                # And to decide what to change, use heuristics?
                
                # Let's try:
                # 1. Verify full net.
                # 2. Try removing last comp. Verify prefix.
                # 3. If valid, pop.
                # 4. If not, try removing second to last? No, too complex.
                
                pass
                
        # Let's implement a simpler, robust removal:
        # We will maintain `states` as the result of `net` applied to `initial`.
        # We will try to remove comparators from the *end* only.
        # If we remove the last one, the new `states` is just `states` before the last step.
        # So we need to store the history of states?
        # Yes, `history[k]` = state after applying `net[k]`.
        # Then removing `net[-1]` means `states = history[-2]`.
        # And we check if `states` is sorted.
        # If yes, we pop `net[-1]` and drop `history[-1]`.
        # This is O(1) per removal check (just an is_sorted check)!
        
        # So:
        # 1. Build `history` and `net`.
        # 2. While possible:
        #    a. Check if `history[-2]` is sorted. (If len>1)
        #    b. If yes, pop `net[-1]`, pop `history[-1]`.
        #    c. If no, break? Or try to remove from other positions?
        #       Removing from other positions invalidates `history`.
        
        # This "remove from end" strategy is very fast.
        # It will remove all redundant comparators at the end.
        
        # Can we do more?
        # We can try to *swap* comparators to bring candidates to the end.
        # Or, we can accept this simple result.
        
        # Let's implement this fast end-removal first.
        
        # Re-implementation:
        
        # 1. Build network and history
        # net = list of comps
        # history = list of state arrays
        
        # But we don't want to store all history if it takes too much memory.
        # 153 * 4.7MB is ~700MB.
        # Let's check if we can do it without storing all.
        # If we only remove from the end, we just need the state *before* the last step.
        # Which is `states_prev`.
        
        # Algorithm:
        # states = apply(net[0:-1], initial)
        # if is_sorted(states):
        #    pop net[-1]
        #    repeat
        # else:
        #    break
        
        # This requires recomputing `states = apply(net[0:-1], initial)` every time?
        # No, if we pop, the new `net[0:-1]` is the old `net[0:-2]`.
        # So we need the state of the *prefix*.
        
        # So we need the history anyway, or we recompute.
        # Recomputing the prefix is O(len).
        
        # Let's just store the history. 700MB should be fine.
        
        # Wait, I can optimize memory.
        # I only need `history[-2]` to check removal of `history[-1]`.
        # But if I pop, the new `history[-2]` is the old `history[-3]`.
        # So I need the whole history.
        
        # Let's try to limit memory.
        # If len(net) > 200, we might run out of memory?
        # 153 is small.
        
        # Let's go with this.
        
        # Re-do the function logic.
        
        pass

    # Actually, let's look at the "Local Candidate Validation Feedback".
    # "empty" reason.
    # My previous code might have been syntactically correct but perhaps I didn't return the right thing?
    # Or maybe the code was too slow?
    # The seed is 153.
    # If I just return the seed, score is 1.0.
    # If I remove some, score > 1.0.
    
    # Let's try to generate a better network using a different method.
    # What if I use a known optimal network for N=18 if available?
    # No, I don't have it.
    
    # Let's try a simple local search on the comparators.
    # Start with OET.
    # Randomly mutate (change wire, add/remove) and verify.
    # Keep if better.
    
    # Verification is fast if we use bit-parallel.
    # 153 comparators, 262144 inputs.
    # Vectorized apply: 153 * (262144 ops) ~ 40M ops.
    # ~0.1s per verification.
    # 60s budget -> 600 verifications.
    
    # Strategy:
    # 1. Start with OET.
    # 2. Loop:
    #    a. Try to remove a random comparator. Verify.
    #    b. If valid, keep.
    #    c. If invalid, revert.
    #    d. Try to change a random comparator (i,j) to (i',j). Verify.
    #    e. If valid and same size (or smaller?), keep.
    #    f. If invalid, revert.
    
    # This is a simple hill climbing.
    
    # Let's implement this.
    
    net = generate_oo_network()
    states = initial_states.copy()
    
    # Verify
    apply_network(states, net)
    if not is_sorted(states):
        return net # Should not happen
    
    best_net = net[:]
    
    # Randomize
    random.seed(42)
    
    while time.time() < deadline:
        changed = False
        # Try removals
        for _ in range(10): # 10 removal trials
            if len(net) == 0:
                break
            idx = random.randint(0, len(net) - 1)
            comp = net[idx]
            
            # Save state
            # To check if removing comp is valid, we need to apply net without comp.
            # This is expensive.
            # We can just apply net without comp to initial_states?
            # Or apply the change to current state?
            # If we remove idx, the state changes for all subsequent comparators.
            # So we must re-verify from scratch or simulate carefully.
            
            # Re-verify from scratch is safe and O(len).
            # 10 * 0.1s = 1s.
            
            new_net = net[:]
            new_net.pop(idx)
            
            s = initial_states.copy()
            apply_network(s, new_net)
            if is_sorted(s):
                net = new_net
                states = s
                changed = True
                best_net = net[:]
                break # Restart loop? Or continue?
        
        if not changed and time.time() > deadline - 10:
            break
            
        # Try mutations
        for _ in range(10):
            if len(net) == 0:
                break
            idx = random.randint(0, len(net) - 1)
            i, j = net[idx]
            
            # Mutate j
            new_j = random.randint(i + 1, N - 1)
            if new_j == j:
                continue
            new_net = net[:]
            new_net[idx] = (i, new_j)
            
            s = initial_states.copy()
            apply_network(s, new_net)
            if is_sorted(s):
                # Check if it's "better"?
                # Same size. So no improvement in count.
                # But might enable future removals.
                net = new_net
                states = s
                changed = True
                break
    
    # Final cleanup: try to remove from end
    # ... (reuse the fast removal logic if we have time)
    
    return best_net

# Let's refine the "fast removal from end" and integrate it.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 18
    size = 1 << N
    k_range = np.arange(size, dtype=np.uint64)
    initial_states = np.empty((N, size), dtype=np.uint8)
    for j in range(N):
        initial_states[j] = ((k_range >> j) & 1).astype(np.uint8)

    def is_sorted(states):
        return not np.any(states[:-1] < states[1:])

    def apply_network(states, network):
        for i, j in network:
            si = states[i].copy()
            sj = states[j].copy()
            states[i] = si & sj
            states[j] = si | sj
        return states

    def generate_oo_network():
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))
        return network

    net = generate_oo_network()
    
    # Verify
    states = initial_states.copy()
    apply_network(states, net)
    if not is_sorted(states):
        return net

    best_net = net[:]
    best_count = len(net)

    # Phase 1: Greedy removal from the end (fast)
    # We maintain `prefix_state` = state after applying net[0:-1]
    # This is hard to maintain without full history.
    # Let's just do a few passes of full verification and random removal.
    
    random.seed(42)
    
    # Time management
    while time.time() < deadline - 5:
        # Try to remove a random comparator
        found_removal = False
        for _ in range(5):
            if len(net) <= best_count:
                break
            idx = random.randint(0, len(net) - 1)
            new_net = net[:]
            new_net.pop(idx)
            
            s = initial_states.copy()
            apply_network(s, new_net)
            if is_sorted(s):
                net = new_net
                states = s
                best_net = net[:]
                best_count = len(net)
                found_removal = True
                break
        
        if not found_removal:
            # Try mutation
            if len(net) > 0:
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                new_j = random.randint(i + 1, N - 1)
                if new_j == j:
                    continue
                new_net = net[:]
                new_net[idx] = (i, new_j)
                
                s = initial_states.copy()
                apply_network(s, new_net)
                if is_sorted(s):
                    net = new_net
                    states = s
                    # Only update best if size is smaller?
                    # Mutation keeps size same. So best_count doesn't change.
                    # But it might help future removals.
                    continue

    # Final pass: remove from end if possible
    # We can try to remove the last comparator repeatedly
    while len(net) > 0:
        if time.time() > deadline:
            break
        new_net = net[:]
        new_net.pop()
        s = initial_states.copy()
        apply_network(s, new_net)
        if is_sorted(s):
            net = new_net
            states = s
            best_net = net[:]
            best_count = len(net)
        else:
            break

    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Result: {len(net)} comparators")
