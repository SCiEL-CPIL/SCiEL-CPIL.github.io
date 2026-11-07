import struct

def update(state, item):
    # 512 counters, 2 bytes each
    p = 1035493289
    idx = (item * p) % 512
    offset = idx * 2
    
    # Copy state to mutable bytearray
    ba = bytearray(state)
    
    # Unpack uint16 at offset
    val = struct.unpack_from('H', ba, offset)[0]
    
    # Increment
    val += 1
    
    # Pack back
    struct.pack_into('H', ba, offset, val)
    
    return bytes(ba)

def query(state, item):
    p = 1035493289
    idx = (item * p) % 512
    offset = idx * 2
    
    # Unpack uint16 at offset
    val = struct.unpack_from('H', state, offset)[0]
    
    return val
