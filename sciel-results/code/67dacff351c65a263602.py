                candidates = set(blocks_containing_subset[uncovered_bits[0]])
                
                for b in uncovered_bits[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
