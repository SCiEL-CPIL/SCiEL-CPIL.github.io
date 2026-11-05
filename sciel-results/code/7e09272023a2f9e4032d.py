<<<<<<< SEARCH
                    # Try to find 1 block that covers all uncovered_bits
                    candidates = None
                    for bit in list(uncovered_bits)[:4]:  # Limit for speed
                        blocks_with_bit = set(subset_to_blocks[bit])
                        if candidates is None:
                            candidates = blocks_with_bit
                        else:
                            candidates &= blocks_with_bits
=======
                    # Try to find 1 block that covers all uncovered_bits
                    candidates = None
                    for bit in list(uncovered_bits)[:4]:  # Limit for speed
                        blocks_with_bit = subset_to_blocks_sets[bit]
                        if candidates is None:
                            candidates = set(blocks_with_bit)
                        else:
                            candidates &= blocks_with_bit
>>>>>>> REPLACE
