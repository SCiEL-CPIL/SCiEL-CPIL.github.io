import struct

# State layout:
# 400 items.
# Max count is 1024, which is 2^10. So 11 bits are needed to represent 0..1024.
# We pack 3 items per 32-bit word (11 * 3 = 33 bits, we waste 1 bit).
# Total words needed: ceil(400 / 3) = 134 words.
# Total bytes: 134 * 4 = 536 bytes.
# This is well within the 1024 byte limit.
# This allows us to have EXACT counts for all items, not just "hot" ones.
# This eliminates all collision errors from CM sketch.

ITEMS_PER_WORD = 3
BITS_PER_ITEM = 11
MASK = (1 << BITS_PER_ITEM) - 1  # 2047

NUM_WORDS = (400 + ITEMS_PER_WORD - 1) // ITEMS_PER_WORD

def _get_item_indices(item):
    """Returns (word_index, bit_offset) for the item."""
    return item // ITEMS_PER_WORD, (item % ITEMS_PER_WORD) * BITS_PER_ITEM

def update(state, item):
    # Convert state to bytearray for mutable operations
    buf = bytearray(state)
    
    word_idx, bit_off = _get_item_indices(item)
    
    # Read the 32-bit word
    # struct.unpack_from('<I', buf, word_idx * 4)
    val = struct.unpack_from('<I', buf, word_idx * 4)[0]
    
    # Increment the specific item's count
    # Mask to isolate the field
    field_mask = MASK << bit_off
    current_val = (val & field_mask) >> bit_off
    new_val = current_val + 1
    
    # Clear the field and set the new value
    val = (val & ~field_mask) | (new_val << bit_off)
    
    # Write back
    struct.pack_into('<I', buf, word_idx * 4, val)
    
    return bytes(buf)

def query(state, item):
    word_idx, bit_off = _get_item_indices(item)
    
    val = struct.unpack_from('<I', state, word_idx * 4)[0]
    
    field_mask = MASK << bit_off
    count = (val & field_mask) >> bit_off
    
    return float(count)
