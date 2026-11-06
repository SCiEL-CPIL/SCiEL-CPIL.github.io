# Inside construct, after adding first_block:
# Add second block fixed by symmetry
second_block = (0, 1, 2, 3, 6, 7)
# Check if valid (it should be, as it shares 4 points with first)
# First: 0,1,2,3,4,5. Second: 0,1,2,3,6,7. Intersection: 0,1,2,3 (size 4). Valid.
blocks.append(second_block)
covered_bits |= block_bitset[second_block]

# Then backtrack for the rest.
