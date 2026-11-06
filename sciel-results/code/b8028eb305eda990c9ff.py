<<<<<<< SEARCH
def construct():
    """
    Return a list of (i, j) comparator pairs that sorts all 16-input sequences.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Start with the known 60-comparator network
    network = list(_KNOWN_60_NETWORK)
    
    # Verify the initial network (should be valid)
    if not _verify_network_fast(network):
        # Fallback: use a simpler but valid network if the known one is broken
        # Generate a simple odd-even transposition network (120 comparators)
        # But first, let's try a bitonic sort
        network = _generate_bitonic_network(N)
        if not _verify_network_fast(network):
            # Last resort: odd-even transposition
            network = _generate_oe_transposition(N)
    
    # Try local search to reduce further
    if (time.time() - start_time) < deadline - 10:
        network = _local_search(network, max_time=deadline - time.time() - 5)
    
    # Final verification
    if not _verify_network_fast(network):
        # Return the known valid network if local search broke something
        return list(_KNOWN_60_NETWORK)
    
    return network
=======
def construct():
    """
    Return a list of (i, j) comparator pairs that sorts all 16-input sequences.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Start with a generated valid Bitonic Sort network (111 comparators for N=16)
    network = _generate_bitonic_network(N)
    
    # Verify the initial network
    if not _verify_network_fast(network):
        # Fallback to Odd-Even Transposition if Bitonic fails (unlikely)
        network = _generate_oe_transposition(N)
        if not _verify_network_fast(network):
            # If both fail, return an empty list? No, must sort.
            # This should not happen.
            raise RuntimeError("Failed to generate a valid sorting network")

    # Try local search to reduce further
    if (time.time() - start_time) < deadline - 10:
        network = _local_search(network, max_time=deadline - time.time() - 5)
    
    # Final verification
    if not _verify_network_fast(network):
        # If local search broke it, revert to the generated valid network
        network = _generate_bitonic_network(N)
        
    # Ensure all comparators are valid (i < j)
    # The generators ensure this, but local search only removes, so it should be safe.
    # However, we do a final sanity check.
    valid_network = []
    for i, j in network:
        if 0 <= i < j < N:
            valid_network.append((i, j))
            
    return valid_network
>>>>>>> REPLACE
