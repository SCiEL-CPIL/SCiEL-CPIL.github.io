"""Exact per-item counting using 10-bit packed counters in 500 bytes of the 1024-byte state."""

# Each item (0..399) gets 10 bits => 4000 bits = 500 bytes.
# We store counts in a bytearray-backed buffer inside the state.
# The state is 1024 bytes; we use the first 500 for counters, rest is padding (zeroes).

def update(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: new state as bytes of length 1024
    """
    # Work on a mutable copy of the first 500 bytes
    buf = bytearray(state[:500])
    
    # Compute bit position for this item's counter
    # item i uses bits [i*10, i*10+9]
    bit_offset = item * 10
    
    # Convert to byte offset and bit offset within byte
    byte_idx = bit_offset // 8
    bit_idx = bit_offset % 8
    
    # Read current 10-bit value
    # We need to read 10 bits starting at bit_idx in byte_idx
    # This may span 2 or 3 bytes
    # Let's read the relevant bytes
    # 10 bits starting at bit position bit_idx in byte byte_idx
    # The bits span from global bit bit_offset to bit_offset+9
    
    # Extract 10-bit value
    # We'll read up to 3 bytes to be safe
    val = 0
    for b in range(3):
        cur_byte = byte_idx + b
        if cur_byte >= 500:
            break
        # Number of bits to take from this byte
        bits_in_byte = 8
        if b == 0:
            bits_in_byte = 8 - bit_idx
        elif b == 2:
            # Only need remaining bits
            pass
        
        # Shift and mask
        shift = b * 8 - bit_idx
        mask = 0xFF
        if b == 0:
            mask = (0xFF >> bit_idx)  # Lower (8 - bit_idx) bits
        elif b == 2:
            # We need at most 10 - (bits already taken) bits
            pass
        
        # Simpler approach: just read the full bytes and extract properly
        pass
    
    # Let me use a cleaner extraction method
    # Read the 10-bit value by reading the necessary bytes
    start_byte = bit_offset // 8
    start_bit = bit_offset % 8
    
    # We need 10 bits, so we might need up to 3 bytes
    # Read bytes from start_byte
    raw = 0
    for i in range(3):
        if start_byte + i < 500:
            raw |= (buf[start_byte + i] << (8 * i))
    
    # Extract 10 bits starting at start_bit
    count = (raw >> start_bit) & 0x3FF  # 10 bits
    
    # Increment count
    count += 1
    
    # Write back the new 10-bit value
    # Clear the old 10 bits
    mask = 0x3FF << start_bit
    # Clear in the buffer
    for i in range(3):
        if start_byte + i < 500:
            byte_mask = mask & (0xFF << (8 * i))
            buf[start_byte + i] &= ~byte_mask
    
    # Set the new 10 bits
    new_val = count << start_bit
    for i in range(3):
        if start_byte + i < 500:
            byte_mask = new_val & (0xFF << (8 * i))
            buf[start_byte + i] |= byte_mask
    
    # Reconstruct state: first 500 bytes are buf, rest is zero
    return bytes(buf) + b'\x00' * 524


def query(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate (int)
    """
    buf = state[:500]
    
    bit_offset = item * 10
    start_byte = bit_offset // 8
    start_bit = bit_offset % 8
    
    raw = 0
    for i in range(3):
        if start_byte + i < 500:
            raw |= (buf[start_byte + i] << (8 * i))
    
    count = (raw >> start_bit) & 0x3FF
    return count
